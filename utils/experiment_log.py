
from dataclasses import fields, is_dataclass
import click
from utils.echo import log_info, log

def log_config_params(cfg):
    def print_dataclass(dc, prefix=""):
        for f in fields(dc):
            value = getattr(dc, f.name)
            field_name = f"{prefix}{f.name}"
            if is_dataclass(value):
                print_dataclass(value, prefix=f"{field_name}.")
            else:
                log(click.style(f"{field_name}:", bold=True), nl=False)
                log(f" {value}")
    log_info("Training configuration summary:")
    log("-" * 60)
    print_dataclass(cfg)
    log("="*60 + "\n")