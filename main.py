import torch
from jäger_bomb_trainer import JägerBombTrainer
from torchvision.transforms import v2 as T
from types import SimpleNamespace
import click
from utils import load_config
from training_factory import TrainingFactory
from freezer import Freezer
from experiment_logger import ExperimentLogger

@click.command()
@click.option("--config", default="schema_template.yaml", help="Path to YAML config file")
def main(config):
    experiment_cfg = load_config(config)
    ExperimentLogger.log_config_params(experiment_cfg)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.set_default_device(device)

    factory = TrainingFactory(experiment_cfg)
    trainerCfg, trainerState = factory.create()
    torch_model = trainerState.model.model
    torch_model.args = SimpleNamespace(box=15, cls=0.5, dfl=2.25)

    freezer = Freezer(torch_model)
    freezer.freeze_backbone_layers(torch_model, experiment_cfg.freeze.backbone_layers)
    
    if experiment_cfg.freeze.dfl:
        freezer.freeze_dfl_conv_weights()

    trainer = JägerBombTrainer(trainerCfg, trainerState)
    trainer.train()

if __name__ == "__main__":
    main()