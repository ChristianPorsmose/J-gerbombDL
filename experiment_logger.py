from configs import ExperimentConfig
import click

class ExperimentLogger:
    @staticmethod
    def log_config_params(cfg : ExperimentConfig):
        click.secho("[INFO] Training configuration summary:", fg="blue", bold=True)
        click.echo("-" * 60)
        click.echo(f"Experiment Name: {cfg.experiment_name}")
        click.echo(f"Model: {cfg.model}")
        click.echo(f"Training Data Path: {cfg.paths.train}")
        click.echo(f"Validation Data Path: {cfg.paths.val}")
        click.echo(f"Test Data Path: {cfg.paths.test}")
        click.echo(f"Batch Size: {cfg.training.batch_size}")
        click.echo(f"Epochs: {cfg.training.epochs}")
        click.echo(f"Learning Rate: {cfg.optimizer.lr}")
        click.echo(f"Optimizer Type: {cfg.optimizer.type}")
        click.echo(f"Learning Rate Scheduler: {cfg.optimizer.lr_scheduler}")
        click.echo(f"Weight Decay: {cfg.optimizer.weight_decay}")
        click.echo(f"Momentum: {cfg.optimizer.momentum}")
        click.echo(f"Freeze Backbone Layers: {cfg.freeze.backbone_layers}")
        click.echo(f"Freeze DFL: {cfg.freeze.dfl}")
        click.echo(f"Data Augmentation: {cfg.augmentation}")
        click.echo(f"Dataset Size: {cfg.dataset_size}")
        click.echo(f"Use EMA: {cfg.use_ema}")
        click.echo(f"Loss Type: {cfg.loss_type}")
        click.echo("="*60 + "\n")
