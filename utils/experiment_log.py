
from dataclasses import fields, is_dataclass
import click

def log_config_params(cfg):
    def print_dataclass(dc, prefix=""):
        for f in fields(dc):
            value = getattr(dc, f.name)
            field_name = f"{prefix}{f.name}"
            if is_dataclass(value):
                print_dataclass(value, prefix=f"{field_name}.")
            else:
                click.echo(click.style(f"{field_name}:", bold=True), nl=False)
                click.echo(f" {value}")
    click.secho("[INFO] Training configuration summary:", fg="blue", bold=True)
    click.echo("-" * 60)
    print_dataclass(cfg)
    click.echo("="*60 + "\n")