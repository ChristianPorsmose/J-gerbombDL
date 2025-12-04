import click

_colors = {
    "debug": "bright_cyan",
    "info": "blue",
    "success": "green",
    "warning": "yellow",
    "error": "red",
    "critical": "bright_red"
}

for level, color in _colors.items():
    globals()[f"log_{level}"] = lambda msg, lvl=level, c=color: click.secho(f"[{lvl.upper()}] {msg}", fg=c)

def log(message :str):
    click.echo(message)
