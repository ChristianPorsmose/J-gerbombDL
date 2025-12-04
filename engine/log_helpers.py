

import click
from engine.data import LossComponent

def log_loss(epoch : int , loss : LossComponent, header : str = ""):
    click.echo(
        f"{header}, " 
        f"Epoch {epoch}, " if epoch else ""
        f"Box: {loss.box:.4f}, "
        f"Cls: {loss.cls:.4f}, "
        f"DFL: {loss.dfl:.4f}, "
        f"Spatial: {loss.spatial:.4f}, " if loss.spatial > 0 else ""
        f"Total: {loss.total():.4f}"
    )
