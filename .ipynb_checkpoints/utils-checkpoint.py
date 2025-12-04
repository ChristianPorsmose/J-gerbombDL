import numpy as np
from pathlib import Path
import torch
import os
import math
from pathlib import Path

from torchvision.transforms import v2 as T
from ultralytics.utils.loss import v8DetectionLoss
import random
import tempfile
import os
from typing import List
import click
    

def xywh_to_xyxy(bboxes: torch.Tensor, img_w: int, img_h: int) -> torch.Tensor:
    """Convert bboxes from normalized xywh to pixel xyxy format."""
    if bboxes.numel() == 0:
        return bboxes
    
    # bboxes: [N, 4] in format [x_center, y_center, width, height] normalized
    x_center = bboxes[:, 0] * img_w
    y_center = bboxes[:, 1] * img_h
    width = bboxes[:, 2] * img_w
    height = bboxes[:, 3] * img_h
    
    x1 = x_center - width / 2
    y1 = y_center - height / 2
    x2 = x_center + width / 2
    y2 = y_center + height / 2
    
    return torch.stack([x1, y1, x2, y2], dim=1)

def unfreeze_all_layers(torch_model):
    """Unfreeze all model parameters for training."""
    for _, param in torch_model.named_parameters():
        param.requires_grad = True
    click.secho("[SUCCESS] All model layers unfrozen and ready for training", fg="green")

def freeze_backbone_layers(torch_model, backbone_to_freeze):
    """Freeze the backbone layers of the model (first N layers before detection head)."""
    frozen_count = 0
    if backbone_to_freeze is None:
        click.echo("No backbone_to_freeze specified, skipping freezing backbone layers.")
        return
    # Cap freezing to layer indices 0..10 so model.11+ are never frozen (detection head)
    if backbone_to_freeze >= 0:
        max_to_freeze = min(int(backbone_to_freeze), 10)

    for name, param in torch_model.named_parameters():
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



def create_experiment_train_path(params, train_data_path):
    with open(train_data_path, 'r') as f:
        all_train_paths = [line.strip() for line in f.readlines()]
        
    desired_size = int(params["dataset_size"]*len(all_train_paths))
        
    if desired_size < len(all_train_paths):
            # Create NESTED subset: shuffle once with fixed seed, then take first N
            # This ensures dataset_size=20 ⊂ dataset_size=40 ⊂ dataset_size=60, etc.
        random.seed(42)  # Fixed seed for reproducibility across all experiments
        all_train_paths_shuffled = all_train_paths.copy()
        random.shuffle(all_train_paths_shuffled)
            
            # Take first N paths (ensures nesting property)
        sampled_paths = all_train_paths_shuffled[:desired_size]
            
            # Create temporary file in the SAME DIRECTORY as original file
            # This is critical because the dataset prepends "../" and looks for labels/ relative to the file location
        orig_dir = os.path.dirname(train_data_path)
        temp_file = tempfile.NamedTemporaryFile(
                mode='w', 
                delete=False, 
                suffix='.txt',
                dir=orig_dir  # Create temp file in same directory as original
            )
        for path in sampled_paths:
            temp_file.write(path + '\n')
        temp_file.close()
        train_data_path = temp_file.name
            
        click.secho(
            f"[INFO] Dataset size limiting: Using {desired_size}/{len(all_train_paths)} training images (NESTED subset)", 
            fg="blue"
        )
        click.secho(f"[INFO] Temporary file created: {train_data_path}", fg="blue")
    else:
        click.secho(f"[INFO] Dataset size: Using all {len(all_train_paths)} training images", fg="blue")
    return train_data_path

def create_transform_list(augmentation_mode):
    if augmentation_mode == "none":
        return [
            LetterBoxTransform(new_shape=(640, 640)),
            T.ToDtype(torch.float32, scale=True)
        ]
    if augmentation_mode == "geometric":
        return [
            LetterBoxTransform(new_shape=(640, 640)),
            T.RandomHorizontalFlip(p=0.5),
            T.RandomVerticalFlip(p=0.5),
            T.RandomPerspective(distortion_scale=0.2, p=0.5, fill=114),
            T.RandomRotation(degrees=(-15, 15), expand=False, fill=114),
            T.ToDtype(torch.float32, scale=True)
        ]
    if augmentation_mode == "light":
        return [
            LetterBoxTransform(new_shape=(640, 640)),
            T.ColorJitter(brightness=0.5, saturation=0.4, hue=0.3),
            T.ToDtype(torch.float32, scale=True)
        ]
    return [
            LetterBoxTransform(new_shape=(640, 640)),
            T.RandomHorizontalFlip(p=0.5),
            T.RandomVerticalFlip(p=0.5),
            T.RandomPerspective(distortion_scale=0.2, p=0.5,fill=114),
            T.RandomRotation(degrees=(-15, 15), expand=False, fill=114),
            T.ColorJitter(brightness=0.5, saturation=0.4, hue=0.3),
            T.ToDtype(torch.float32, scale=True)
        ]

def create_loss_func(torch_model, loss_type):
    if loss_type == "spatial_consistency":
        click.secho("[INFO] Using spatial consistency loss (JägerBombLoss)", fg="blue")
        return JägerBombLoss(torch_model, lamda_rate=1)
    click.secho("[INFO] Using standard YOLO loss (v8DetectionLoss)", fg="blue")
    return v8DetectionLoss(torch_model)

def create_scheduler(params, lr, optimizer, lr_scheduler_type):
    if lr_scheduler_type == "cosine":
        def one_cycle_lr(epoch):
            """Cosine learning rate schedule from 1.0 to lrf over epochs."""
            lrf = 0.01  # final learning rate factor (1% of initial)
            # Cosine annealing: starts at 1.0, ends at lrf
            return lrf + (1 - lrf) * 0.5 * (1 + math.cos(math.pi * epoch / params["epochs"]))
        
        click.echo(f"Learning rate scheduler: cosine decay from {lr:.6f} to {lr * 0.01:.6f} over {params['epochs']} epochs")
        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=one_cycle_lr)

    click.echo(f"Learning rate: fixed at {lr:.6f}")
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda epoch: 1.0)

def build_optimizer_params(torch_model):
    g = [], [], []  # parameter groups: [weights with decay, weights without decay, biases]
    bn = tuple(v for k, v in torch.nn.__dict__.items() if "Norm" in k)  # normalization layers
    
    for module_name, module in torch_model.named_modules():
        for param_name, param in module.named_parameters(recurse=False):
            fullname = f"{module_name}.{param_name}" if module_name else param_name
            if "bias" in fullname:  # bias (no decay)
                g[2].append(param)
            elif isinstance(module, bn):  # batch norm weights (no decay)
                g[1].append(param)
            else:  # regular weights (with decay)
                g[0].append(param)
    return g

def create_optimizer(params: dict, param_groups: List, optimizer_type: str, lr: float, weight_decay: float):
    """
    Create an optimizer with separate parameter groups for weights, batchnorm, and biases.
    
    Args:
        params (dict): Experiment parameters (may include momentum)
        param_groups (list): [weights_with_decay, batchnorm_no_decay, biases_no_decay]
        optimizer_type (str): 'SGD' or 'AdamW'
        lr (float): Learning rate
        weight_decay (float): Weight decay for regular weights
    """
    weights, bn_no_decay, biases = param_groups
    optimizer_type = optimizer_type.upper()

    if optimizer_type == "SGD":
        momentum = params.get("momentum", 0.0)
        use_nesterov = momentum > 0
        optimizer = torch.optim.SGD(biases, lr=lr, momentum=momentum, nesterov=use_nesterov, weight_decay=0.0)
    elif optimizer_type == "ADAMW":
        momentum = params.get("momentum", 0.9)
        optimizer = torch.optim.AdamW(biases, lr=lr, betas=(momentum, 0.999), weight_decay=0.0)
    else:
        raise ValueError(f"Unsupported optimizer type: {optimizer_type}")

    optimizer.add_param_group({"params": weights, "weight_decay": weight_decay})
    optimizer.add_param_group({"params": bn_no_decay, "weight_decay": 0.0})

    click.secho(f"[SUCCESS] Optimizer created: {optimizer_type} | lr={lr}, weight_decay={weight_decay}, momentum={momentum}", fg="green")
    click.echo(f"Parameter groups: {len(weights)} weights(decay), {len(bn_no_decay)} batchnorm(no decay), {len(biases)} biases(no decay)")

    return optimizer
