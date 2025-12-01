import yaml
import torch
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from ultralytics.models import YOLO
import os
import math
from pathlib import Path

from jäger_bomb_dataset import JägerBombDataset
from jäger_bomb_loss import JägerBombLoss
from jäger_bomb_trainer import JägerBombTrainer
from configs import TrainingConfig, SaveConfig
from torchvision.transforms import v2 as T
from types import SimpleNamespace
from ultralytics.utils.loss import v8DetectionLoss
from letter_box_transform import LetterBoxTransform
from yolo_compose import YOLOCompose

def collate_fn(batch):
    images, targets = zip(*batch)
    padded_targets = pad_sequence(targets, batch_first=True, padding_value=-1)
    images = torch.stack(images, 0)
    return images, padded_targets


def unfreeze_all_layers(torch_model):
    """Unfreeze all model parameters for training."""
    
    #Print the architecture of the model


    for name, param in torch_model.named_parameters():
        param.requires_grad = True
        print("Unfroze model layer:", name)
    print("✅ All model layers unfrozen and ready for training")


def freeze_backbone_layers(torch_model, backbone_to_freeze):
    """Freeze the backbone layers of the model (first N layers before detection head)."""
    frozen_count = 0

    if backbone_to_freeze is None:
        #THrow an error
        print("No backbone_to_freeze specified, skipping freezing backbone layers.")
        return
    # Cap freezing to layer indices 0..10 so model.11+ are never frozen (detection head)
    if backbone_to_freeze >= 0:
        max_to_freeze = min(int(backbone_to_freeze), 10)

    for name, param in torch_model.named_parameters():
        # Expect names like "model.0.conv.weight" -> extract the index after "model."
        if name.startswith("model."):
            rest = name[len("model."):]
            idx_str = rest.split('.', 1)[0]
            try:
                idx = int(idx_str)
            except ValueError:
                continue
            if 0 <= idx <= max_to_freeze:
                param.requires_grad = False
                frozen_count += 1

    print(f"✅ Froze {frozen_count} backbone parameters (model.0..model.{max_to_freeze})")


def freeze_dfl_conv_weights(torch_model):
    """Freeze the weights of dfl.conv layers in the model."""
    found = False

    for name, module in torch_model.named_modules():
        if name.endswith('.dfl.conv'):
            for pname, param in module.named_parameters(recurse=False):
                if pname == 'weight':
                    param.requires_grad = False
                    print(f"✅ Froze DFL convolution weights at path: {name}")
                    found = True
    if not found:
        print("❌ Error: Could not locate the DFL convolution module.")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Train Jäger Bomb Detection Model")
    parser.add_argument("--config", type=str, default="setup.yaml", 
                        help="Path to config file (default: setup.yaml)")
    parser.add_argument("--config-dict", type=str, default=None,
                        help="Config name from experiment_configs (e.g., PHASE1A_SGD_STANDARD)")
    args = parser.parse_args()

    # Load config from dict or YAML file
    if args.config_dict:
        import experiment_configs
        params = getattr(experiment_configs, args.config_dict)
        print(f"📦 Loaded config: {args.config_dict}")
    else:
        with open(args.config, "r") as f:
            params = yaml.safe_load(f)
        print(f"📦 Loaded config from: {args.config}")
    
    # Get experiment name from config parameter
    experiment_name = params.get("experiment_name")
    if experiment_name:
        print(f"🧪 Running experiment: {experiment_name}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.set_default_device(device)

    model = YOLO( params["model"], task="detect").load('yolo11n.pt')
    
    #model.names = {1:"cup", 0:"shot"}
    torch_model = model.model
    torch_model.to(device)

    unfreeze_all_layers(torch_model)
    
    # Freeze backbone if requested in config
    layers_to_freeze = params.get("freeze_backbone_layers", None)
    print(f"Layers to freeze in backbone: {layers_to_freeze}")
    freeze_backbone_layers(torch_model, layers_to_freeze)
    
    # Freeze DFL weights if requested in config
    if params.get("freeze_dfl", False):
        freeze_dfl_conv_weights(torch_model)

    # Now add the hyperparameters as attributes
    torch_model.args = SimpleNamespace(box=15, cls=0.5, dfl=2.25)

    # Build optimizer with proper parameter groups (like Ultralytics does)
    g = [], [], []  # parameter groups: [weights with decay, weights without decay, biases]
    bn = tuple(v for k, v in torch.nn.__dict__.items() if "Norm" in k)  # normalization layers
    
    for module_name, module in torch_model.named_modules():
        for param_name, param in module.named_parameters(recurse=False):
            fullname = f"{module_name}.{param_name}" if module_name else param_name
            if "bias" in fullname:  # bias (no decay)
                g[2].append(param)
            elif isinstance(module, bn):  # batch norm weights (no decay)
                g[1].append(param)
            else:  # regular weights (with decay)
                g[0].append(param)
    
    # Setup optimizer based on config
    optimizer_type = params.get("optimizer", "AdamW")
    lr = params.get("lr", 0.001667)
    weight_decay = params.get("weight_decay", 0.0005625)
    
    if optimizer_type == "SGD":
        momentum = params.get("momentum", 0.0)  # Default to 0 for vanilla SGD
        use_nesterov = momentum > 0  # Only use Nesterov if momentum is specified
        optimizer = torch.optim.SGD(g[2], lr=lr, momentum=momentum, nesterov=use_nesterov, weight_decay=0.0)
        optimizer.add_param_group({"params": g[0], "weight_decay": weight_decay})  # weights with decay
        optimizer.add_param_group({"params": g[1], "weight_decay": 0.0})  # batch norm without decay
        if momentum > 0:
            print(f"optimizer: SGD(lr={lr}, momentum={momentum}, nesterov={use_nesterov}) with {len(g[1])} weight(decay=0.0), {len(g[0])} weight(decay={weight_decay}), {len(g[2])} bias(decay=0.0)")
        else:
            print(f"optimizer: SGD(lr={lr}) with {len(g[1])} weight(decay=0.0), {len(g[0])} weight(decay={weight_decay}), {len(g[2])} bias(decay=0.0)")
    elif optimizer_type == "AdamW":
        momentum = params.get("momentum", 0.9)
        optimizer = torch.optim.AdamW(g[2], lr=lr, betas=(momentum, 0.999), weight_decay=0.0)
        optimizer.add_param_group({"params": g[0], "weight_decay": weight_decay})  # weights with decay
        optimizer.add_param_group({"params": g[1], "weight_decay": 0.0})  # batch norm without decay
        print(f"optimizer: AdamW(lr={lr}, betas=({momentum}, 0.999)) with {len(g[1])} weight(decay=0.0), {len(g[0])} weight(decay={weight_decay}), {len(g[2])} bias(decay=0.0)")
    else:
        raise ValueError(f"Unsupported optimizer type: {optimizer_type}")
    # TRAINING transforms - augmentation mode based on config
    augmentation_mode = params.get("augmentation", "full")
    
    if augmentation_mode == "none":
        # No augmentation except letterbox
        train_transform_list = [
            LetterBoxTransform(new_shape=(640, 640)),
            T.ToDtype(torch.float32, scale=True)
        ]
    elif augmentation_mode == "geometric":
        # Geometric augmentations only (flips, perspective, rotation)
        train_transform_list = [
            LetterBoxTransform(new_shape=(640, 640)),
            T.RandomHorizontalFlip(p=0.5),
            T.RandomVerticalFlip(p=0.5),
            T.RandomPerspective(distortion_scale=0.2, p=0.5, fill=114),
            T.RandomRotation(degrees=(-15, 15), expand=False, fill=114),
            T.ToDtype(torch.float32, scale=True)
        ]
    elif augmentation_mode == "light":
        # Light-related augmentations only
        train_transform_list = [
            LetterBoxTransform(new_shape=(640, 640)),
            T.ColorJitter(brightness=0.5, saturation=0.4, hue=0.3),

            T.ToDtype(torch.float32, scale=True)
        ]
    else:  # "full" - custom augmentations for disco/club environment
        train_transform_list = [
            LetterBoxTransform(new_shape=(640, 640)),
            T.RandomHorizontalFlip(p=0.5),
            T.RandomVerticalFlip(p=0.5),
            T.RandomPerspective(distortion_scale=0.2, p=0.5,fill=114),
            T.RandomRotation(degrees=(-15, 15), expand=False, fill=114),
            T.ColorJitter(brightness=0.5, saturation=0.4, hue=0.3),
            T.ToDtype(torch.float32, scale=True)
        ]
    
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
        # Read all training image paths
        with open(train_data_path, 'r') as f:
            all_train_paths = [line.strip() for line in f.readlines()]
        
        desired_size = int(params["dataset_size"]*len(all_train_paths))
        
        if desired_size < len(all_train_paths):
            # Create NESTED subset: shuffle once with fixed seed, then take first N
            # This ensures dataset_size=20 ⊂ dataset_size=40 ⊂ dataset_size=60, etc.
            import random
            random.seed(42)  # Fixed seed for reproducibility across all experiments
            all_train_paths_shuffled = all_train_paths.copy()
            random.shuffle(all_train_paths_shuffled)
            
            # Take first N paths (ensures nesting property)
            sampled_paths = all_train_paths_shuffled[:desired_size]
            
            # Create temporary file in the SAME DIRECTORY as original file
            # This is critical because the dataset prepends "../" and looks for labels/ relative to the file location
            import tempfile
            import os
            orig_dir = os.path.dirname(train_data_path)
            temp_file = tempfile.NamedTemporaryFile(
                mode='w', 
                delete=False, 
                suffix='.txt',
                dir=orig_dir  # Create temp file in same directory as original
            )
            for path in sampled_paths:
                temp_file.write(path + '\n')
            temp_file.close()
            train_data_path = temp_file.name
            
            print(f"📊 Dataset size limiting: Using {desired_size}/{len(all_train_paths)} training images (NESTED subset)")
            print(f"📁 Temp file created: {train_data_path}")
        else:
            print(f"📊 Dataset size: Using all {len(all_train_paths)} training images")

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
    
    # Test dataset (no augmentation!)
    test_dataset = JägerBombDataset(params["test_data_path"], transforms=val_transforms)
    test_loader = DataLoader(
        test_dataset,
        batch_size=params["batch_size"],
        shuffle=False,
        collate_fn=collate_fn,
        generator=torch.Generator(device)
    )
    
    # Setup learning rate scheduler
    lr_scheduler_type = params.get("lr_scheduler", "fixed")  # Default to fixed
    
    if lr_scheduler_type == "cosine":
        def one_cycle_lr(epoch):
            """Cosine learning rate schedule from 1.0 to lrf over epochs."""
            lrf = 0.01  # final learning rate factor (1% of initial)
            # Cosine annealing: starts at 1.0, ends at lrf
            return lrf + (1 - lrf) * 0.5 * (1 + math.cos(math.pi * epoch / params["epochs"]))
        
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=one_cycle_lr)
        print(f"Learning rate scheduler: cosine decay from {lr:.6f} to {lr * 0.01:.6f} over {params['epochs']} epochs")
    else:
        # Fixed learning rate (no scheduler)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda epoch: 1.0)
        print(f"Learning rate: fixed at {lr:.6f}")

    #print("Dataset length:", len(train_dataset))
    #print("Testing __getitem__")
    #print(train_dataset[3])

    # Select loss function based on configuration
    loss_type = params.get("loss_type", "standard")
    if loss_type == "spatial_consistency":
        loss_fn = JägerBombLoss(torch_model,lamda_rate=1)
        print("🎯 Using spatial consistency loss (JägerBombLoss)")
    else:
        loss_fn = v8DetectionLoss(torch_model)
        print("📦 Using standard YOLO loss (v8DetectionLoss)")
    
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