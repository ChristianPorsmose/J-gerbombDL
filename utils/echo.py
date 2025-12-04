import click

def log_info(msg: str):
    click.secho(f"[INFO] {msg}", fg="blue")

def log_success(msg: str):
    click.secho(f"[SUCCESS] {msg}", fg="green")

def log_warning(msg: str):
    click.secho(f"[WARNING] {msg}", fg="yellow")

def log_error(msg: str):
    click.secho(f"[ERROR] {msg}", fg="red")


def log_debug(msg : str):
    click.secho(f"[DEBUG] {msg}", fg="cyan")

def log(message: str, *args, **kwargs):
    click.echo(message, *args, **kwargs)
