from engine.data import LossComponent
from utils.echo import log_debug


def log_loss(epoch: int, loss: LossComponent, header: str = ""):
    log_debug(f"{header}, ")
    log_debug(f"Epoch {epoch}, " if epoch is not None else "")
    log_debug(f"Box: {loss.box:.4f}, ")
    log_debug(f"Cls: {loss.cls:.4f}, ")
    log_debug(f"DFL: {loss.dfl:.4f}, ")
    log_debug(f"Spatial: {loss.spatial:.4f}, " if loss.spatial > 0 else "")
    log_debug(f"Total: {loss.total():.4f}")
