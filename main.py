import torch
from engine.jäger_bomb_trainer import JägerBombTrainer
from torchvision.transforms import v2 as T
import click
from utils.utils import load_config
from factory.training_factory import TrainingFactory
from utils.experiment_log import log_config_params
from freezer.freezer import Freezer

@click.command()
@click.option("--config", default="schema_template.yaml", help="Path to YAML config file")
def main(config):
    experiment_cfg = load_config(config)
    log_config_params(experiment_cfg)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.set_default_device(device)

    factory = TrainingFactory(experiment_cfg)
    trainerCfg, trainerState = factory.create()
    torch_model = trainerState.model.model

    freezer = Freezer(torch_model)
    freezer.freeze_backbone_layers(experiment_cfg.freeze.backbone_layers)
    
    if experiment_cfg.freeze.dfl:
        freezer.freeze_dfl_conv_weights()

    trainer = JägerBombTrainer(trainerCfg, trainerState)
    trainer.train()
    trainer.test_best_model()

if __name__ == "__main__":
    main()