
import click
from torch import nn
from utils.echo import log_success, log_error, log

class Freezer:
    def __init__(self, model: nn.Module):
        self.model = model
        self._unfreeze_all_layers()

    def _unfreeze_all_layers(self):
        """Unfreeze all model parameters for training."""
        for _, param in self.model.named_parameters():
            param.requires_grad = True
        log_success("All model layers unfrozen and ready for training")

    def freeze_backbone_layers(self, num_layers: int):
        """Freeze the backbone layers of the model (first N layers before detection head)."""
        frozen_count = 0

        params_to_train = []

        for name, module in self.model.named_children():
            if name == 'model':
                for idx, child in enumerate(module.children()):
                        if (idx < num_layers):
                            for param in child.parameters():
                                param.requires_grad = False
                                frozen_count += 1
        
        for name, param in self.model.named_parameters():
            if param.requires_grad==True:
                params_to_train.append(param)
                            
        log_success(f"Froze {frozen_count} backbone parameters (model.0..model.{num_layers})")
        return params_to_train


    def freeze_dfl_conv_weights(self):
        """Freeze the weights of dfl.conv layers in the model."""
        found = False

        for name, module in self.model.named_modules():
            if name.endswith('.dfl.conv'):
                for pname, param in module.named_parameters(recurse=False):
                    if pname == 'weight':
                        param.requires_grad = False
                        found = True
        log_success(f"Froze DFL convolution weights at path: {name}")
        if not found:
            log_error("Could not locate the DFL convolution module.")