
import click
from torch import nn

class Freezer:
    def __init__(self, model: nn.Module):
        self.model = model
        self._unfreeze_all_layers()

    def _unfreeze_all_layers(self):
        """Unfreeze all model parameters for training."""
        for _, param in self.model.named_parameters():
            param.requires_grad = True
        click.secho("[SUCCESS] All model layers unfrozen and ready for training", fg="green")

    def freeze_backbone_layers(self, num_layers: int):
        """Freeze the backbone layers of the model (first N layers before detection head)."""
        frozen_count = 0
        if num_layers is None:
            click.echo("No backbone_to_freeze specified, skipping freezing backbone layers.")
            return
        # Cap freezing to layer indices 0..10 so model.11+ are never frozen (detection head)
        if num_layers >= 0:
            max_to_freeze = min(int(num_layers), 10)

        for name, param in self.model.named_parameters():
            # Expect names like "model.0.conv.weight" -> extract the index after "model."
            if name.startswith("model."):
                rest = name[len("model."):]
                idx_str = rest.split('.', 1)[0]
                try:
                    idx = int(idx_str)
                except ValueError:
                    continue
                if 0 <= idx <= max_to_freeze:
                    param.requires_grad = False
                    frozen_count += 1

        click.secho(f"[SUCCESS] Froze {frozen_count} backbone parameters (model.0..model.{max_to_freeze})", fg="green")

    def freeze_dfl_conv_weights(self):
        """Freeze the weights of dfl.conv layers in the model."""
        found = False

        for name, module in self.model.named_modules():
            if name.endswith('.dfl.conv'):
                for pname, param in module.named_parameters(recurse=False):
                    if pname == 'weight':
                        param.requires_grad = False
                        found = True
        click.secho(f"[SUCCESS] Froze DFL convolution weights at path: {name}", fg="green")
        if not found:
            click.secho("[ERROR] Could not locate the DFL convolution module.", fg="red")