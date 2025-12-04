import torch
from engine.jäger_bomb_trainer import JägerBombTrainer
from torchvision.transforms import v2 as T
import click
from utils.utils import load_config, create_experiment_train_path
from factory.training_factory import TrainingFactory
from utils.experiment_log import log_config_params
from freezer.freezer import Freezer

@click.command()
@click.option("--config", default="main.yaml", help="Path to YAML config file")
def main(config):
    experiment_cfg = load_config(config)
    log_config_params(experiment_cfg)

    with create_experiment_train_path(experiment_cfg) as temp_train_file:
        experiment_cfg.paths.train = temp_train_file

        device = "cuda" if torch.cuda.is_available() else "cpu"
        torch.set_default_device(device)

        factory = TrainingFactory(experiment_cfg)
        model = factory.create_yolo_model()
        torch_model = model.model

        freezer = Freezer(torch_model)
        params = freezer.freeze_backbone_layers(experiment_cfg.freeze.backbone_layers)

        trainerCfg, trainerState = factory.create(model, params)
        
        if experiment_cfg.freeze.dfl:
            freezer.freeze_dfl_conv_weights()

        trainer = JägerBombTrainer(trainerCfg, trainerState)
        trainer.train()
        trainer.test_best_model()

if __name__ == "__main__":
    main()
