import yaml
import torch
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from ultralytics.models import YOLO
from jäger_bomb_dataset import JägerBombDataset
from jäger_bomb_trainer import JägerBombTrainer
from configs import TrainingConfig, SaveConfig
from torchvision.transforms import v2 as T
from types import SimpleNamespace
from letter_box_transform import LetterBoxTransform
from yolo_compose import YOLOCompose
import experiment_configs
import click
from utils import (
    freeze_backbone_layers,
    unfreeze_all_layers,
    freeze_dfl_conv_weights,
    build_optimizer_params,
    create_optimizer,
    create_scheduler,
    create_loss_func,
    create_transform_list,
    create_experiment_train_path
)

def collate_fn(batch):
    images, targets = zip(*batch)
    padded_targets = pad_sequence(targets, batch_first=True, padding_value=-1)
    images = torch.stack(images, 0)
    return images, padded_targets

# TODO: change config-dicts to actually just be a file for each experiment -> cleaner code
@click.command()
@click.option("--config", default="setup.yaml", help="Path to YAML config file")
@click.option("--config-dict", default=None, help="Config name from experiment_configs (e.g., PHASE1A_SGD_STANDARD)")
def main(config, config_dict):
    if config_dict:
        params = getattr(experiment_configs, config_dict)
        click.secho(f"[INFO] Loaded config: {config_dict}", fg="blue")
    else:
        with open(config, "r") as f:
            params = yaml.safe_load(f)
        click.secho(f"[INFO] Loaded config from {config}", fg="blue")
    
    # TODO : following the other todo, this can just extract the name from the file name
    experiment_name = params.get("experiment_name")
    if experiment_name:
       click.secho(f"[EXPERIMENT] Running experiment: {experiment_name}", fg="cyan", bold=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.set_default_device(device)

    model = YOLO( params["model"], task="detect").load('yolo11n.pt')
    
    torch_model = model.model
    torch_model.to(device)

    unfreeze_all_layers(torch_model)
    
    layers_to_freeze = params.get("freeze_backbone_layers", None)
    print(f"Layers to freeze in backbone: {layers_to_freeze}")
    freeze_backbone_layers(torch_model, layers_to_freeze)
    
    if params.get("freeze_dfl", False):
        freeze_dfl_conv_weights(torch_model)

    torch_model.args = SimpleNamespace(box=15, cls=0.5, dfl=2.25)

    g = build_optimizer_params(torch_model)
   
    optimizer_type = params.get("optimizer", "AdamW")
    lr = params.get("lr", 0.001667)
    weight_decay = params.get("weight_decay", 0.0005625)
    
    optimizer = create_optimizer(params, g, optimizer_type, lr, weight_decay)
    
    augmentation_mode = params.get("augmentation", "full")
    train_transform_list = create_transform_list(augmentation_mode)
    
    train_transforms = YOLOCompose(train_transform_list)
    print(f"Augmentation mode: {augmentation_mode}")
    
    # VALIDATION transforms (no augmentation!)
    val_transforms = YOLOCompose([
        LetterBoxTransform(new_shape=(640, 640)),
        T.ToDtype(torch.float32, scale=True)
    ])

    # Handle dataset size limiting (for Phase 3 experiments)
    # IMPORTANT: Creates NESTED subsets - smaller sizes are subsets of larger sizes
    train_data_path = params["train_data_path"]
    if "dataset_size" in params and params["dataset_size"] is not None:
        train_data_path = create_experiment_train_path(params, train_data_path)

    train_dataset = JägerBombDataset(train_data_path, transforms=train_transforms)
    train_loader = DataLoader(
        train_dataset,
        batch_size=params["batch_size"],
        shuffle=True,
        collate_fn=collate_fn,
        generator=torch.Generator(device)
    )

    val_dataset = JägerBombDataset(params["val_data_path"], transforms=val_transforms)
    val_loader = DataLoader(
        val_dataset,
        batch_size=params["batch_size"],
        shuffle=False,
        collate_fn=collate_fn,
        generator=torch.Generator(device)
    )
    
    test_dataset = JägerBombDataset(params["test_data_path"], transforms=val_transforms)
    test_loader = DataLoader(
        test_dataset,
        batch_size=params["batch_size"],
        shuffle=False,
        collate_fn=collate_fn,
        generator=torch.Generator(device)
    )
    
    lr_scheduler_type = params.get("lr_scheduler", "fixed") 
    scheduler = create_scheduler(params, lr, optimizer, lr_scheduler_type)

    loss_type = params.get("loss_type", "standard")
    loss_fn = create_loss_func(torch_model, loss_type)
    
    cfg = TrainingConfig(
        model=torch_model,
        optimizer=optimizer,
        scheduler=scheduler,
        train_dataloader=train_loader,
        val_dataloader=val_loader,
        test_dataloader=test_loader,
        loss_fn=loss_fn,
        device=device,
        epochs=params["epochs"],
        log_interval=params["log_interval"],
        yolo_model=model,
        use_ema=params.get("use_ema", False),
        freeze_dfl=params.get("freeze_dfl", False),
        experiment_name=experiment_name,
        loss_type=loss_type,
        config_params=params
    )
    
    save_cfg = SaveConfig(
        save_epoch_interval=params["save_interval"],
        save_path=params["save_path"]
    )
    trainer = JägerBombTrainer(cfg, save_cfg)

    trainer.train()



if __name__ == "__main__":
    main()