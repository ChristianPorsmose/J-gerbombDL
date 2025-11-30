import yaml
import torch
from torch.optim import Adam, SGD
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from ultralytics.models import YOLO
import os
import math
from pathlib import Path

from jäger_bomb_dataset import JägerBombDataset
from jäger_bomb_loss import JägerBombLoss
from jäger_bomb_trainer import JägerBombTrainer
from configs import TrainingConfig, SaveConfig
from torchvision.transforms import v2 as T
from types import SimpleNamespace
from ultralytics.utils.loss import v8DetectionLoss

class YOLOCompose:
    """Custom compose that handles both images and bboxes."""
    def __init__(self, transforms):
        self.transforms = transforms
    
    def __call__(self, img, bboxes=None):
        for t in self.transforms:
            if isinstance(t, LetterBoxTransform):
                if bboxes is not None and len(bboxes) > 0:
                    img, bboxes = t(img, bboxes)
                else:
                    img = t(img)
            elif isinstance(t, T.RandomHorizontalFlip) and bboxes is not None and len(bboxes) > 0:
                # Apply horizontal flip to both image and bboxes
                if torch.rand(1) < t.p:
                    img = T.functional.hflip(img)
                    # Flip bbox x-coordinates: x_center_new = 1 - x_center_old
                    bboxes[:, 1] = 1.0 - bboxes[:, 1]  # flip x_center (column 1)
            else:
                # Regular transforms that only affect the image
                img = t(img)
        return img, bboxes

class LetterBoxTransform:
    def __init__(self, new_shape=(640, 640), color=(114, 114, 114)):
        self.new_shape = new_shape  # (H, W)
        self.color = color
        self.scale_ratio = None
        self.pad = None
    
    def __call__(self, img, bboxes=None):
        """
        Apply letterbox to image and optionally transform bboxes.
        
        Args:
            img: tensor [C, H, W]
            bboxes: optional tensor [N, 5] in YOLO format [class, x_center, y_center, width, height] (normalized 0-1)
        
        Returns:
            img: letterboxed image
            bboxes: transformed bboxes (if provided)
        """
        # img is a tensor [C, H, W]
        shape = img.shape[1:]  # current shape [H, W]
        new_shape = self.new_shape
        
        # Scale ratio (new / old)
        r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
        self.scale_ratio = r
        
        # Compute padding
        new_unpad = int(round(shape[0] * r)), int(round(shape[1] * r))
        dh, dw = new_shape[0] - new_unpad[0], new_shape[1] - new_unpad[1]  # wh padding
        
        dh /= 2  # divide padding into 2 sides
        dw /= 2
        
        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        self.pad = (left, top, right, bottom)
        
        if shape != new_unpad:  # resize
            img = T.Resize(new_unpad)(img)
        
        # Add padding
        img = T.Pad([left, top, right, bottom], fill=self.color[0])(img)  # Use first color value for all channels
        
        # Transform bboxes if provided
        if bboxes is not None and len(bboxes) > 0:
            # Bboxes are in normalized format [class, x_center, y_center, width, height]
            # We need to maintain them in normalized format relative to the NEW padded image
            
            # Original image dimensions
            orig_h, orig_w = shape
            
            # New dimensions after padding
            new_h, new_w = new_shape
            
            # Convert from normalized to pixel coordinates in original image
            bboxes_pixel = bboxes.clone()
            bboxes_pixel[:, 1] *= orig_w  # x_center
            bboxes_pixel[:, 2] *= orig_h  # y_center
            bboxes_pixel[:, 3] *= orig_w  # width
            bboxes_pixel[:, 4] *= orig_h  # height
            
            # Apply scaling
            bboxes_pixel[:, 1:5] *= r
            
            # Apply padding offset (only to centers)
            bboxes_pixel[:, 1] += left  # x_center
            bboxes_pixel[:, 2] += top   # y_center
            
            # Normalize to new image size
            bboxes_pixel[:, 1] /= new_w  # x_center
            bboxes_pixel[:, 2] /= new_h  # y_center
            bboxes_pixel[:, 3] /= new_w  # width
            bboxes_pixel[:, 4] /= new_h  # height
            
            return img, bboxes_pixel
        
        return img


def collate_fn(batch):
    images, targets = zip(*batch)
    padded_targets = pad_sequence(targets, batch_first=True, padding_value=-1)
    images = torch.stack(images, 0)
    return images, padded_targets


def unfreeze_all_layers(torch_model):
    """Unfreeze all model parameters for training."""
    
    #Print the architecture of the model


    for name, param in torch_model.named_parameters():
        param.requires_grad = True
        print("Unfroze model layer:", name)
    print("✅ All model layers unfrozen and ready for training")


def freeze_backbone_layers(torch_model):
    """Freeze the backbone layers of the model (first N layers before detection head)."""
    frozen_count = 0
    # Freeze all layers in model.model (backbone)

    #Do not freeze any layer from the detection head (from model.11 onwards)

    for name, param in torch_model.named_parameters():
        # Freeze everything except the detection head (last layers)
        if not any(x in name for x in ['model.11', 'model.12', 'model.13', 'model.14', 'model.15', 'model.16', 'model.17', 'model.18', 'model.19', 'model.20', 'model.21', 'model.22', 'model.23']):
            param.requires_grad = False
            #print("Froze model layer:", name)
            frozen_count += 1
    print(f"✅ Froze {frozen_count} backbone parameters (keeping detection head trainable)")


def freeze_dfl_conv_weights(torch_model):
    """Freeze the weights of dfl.conv layers in the model."""
    found = False

    for name, module in torch_model.named_modules():
        if name.endswith('.dfl.conv'):
            for pname, param in module.named_parameters(recurse=False):
                if pname == 'weight':
                    param.requires_grad = False
                    print(f"✅ Froze DFL convolution weights at path: {name}")
                    found = True
    if not found:
        print("❌ Error: Could not locate the DFL convolution module.")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Train Jäger Bomb Detection Model")
    parser.add_argument("--config", type=str, default="setup.yaml", 
                        help="Path to config file (default: setup.yaml)")
    parser.add_argument("--config-dict", type=str, default=None,
                        help="Config name from experiment_configs (e.g., PHASE1A_SGD_STANDARD)")
    args = parser.parse_args()

    # Load config from dict or YAML file
    if args.config_dict:
        import experiment_configs
        params = getattr(experiment_configs, args.config_dict)
        print(f"📦 Loaded config: {args.config_dict}")
    else:
        with open(args.config, "r") as f:
            params = yaml.safe_load(f)
        print(f"📦 Loaded config from: {args.config}")
    
    # Get experiment name from config parameter
    experiment_name = params.get("experiment_name")
    if experiment_name:
        print(f"🧪 Running experiment: {experiment_name}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.set_default_device(device)

    model = YOLO( params["model"], task="detect").load('yolo11n.pt')
    
    #model.names = {1:"cup", 0:"shot"}
    torch_model = model.model
    torch_model.to(device)

    unfreeze_all_layers(torch_model)
    
    # Freeze backbone if requested in config
    if params.get("freeze_backbone", False):
        freeze_backbone_layers(torch_model)
    
    # Freeze DFL weights if requested in config
    if params.get("freeze_dfl", False):
        freeze_dfl_conv_weights(torch_model)

    # Now add the hyperparameters as attributes
    torch_model.args = SimpleNamespace(box=15, cls=0.5, dfl=2.25)

    # Build optimizer with proper parameter groups (like Ultralytics does)
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
    
    # Use AdamW with proper weight decay setup
    lr = params.get("lr", 0.001667)
    momentum = 0.9
    weight_decay = params.get("weight_decay", 0.0005625)
    optimizer = torch.optim.AdamW(g[2], lr=lr, betas=(momentum, 0.999), weight_decay=0.0)
    optimizer.add_param_group({"params": g[0], "weight_decay": weight_decay})  # weights with decay
    optimizer.add_param_group({"params": g[1], "weight_decay": 0.0})  # batch norm without decay
    
    print(f"optimizer: AdamW(lr={lr}, momentum={momentum}) with parameter groups {len(g[1])} weight(decay=0.0), {len(g[0])} weight(decay={weight_decay}), {len(g[2])} bias(decay=0.0)")

    # TRAINING transforms - augmentation mode based on config
    augmentation_mode = params.get("augmentation", "full")
    
    if augmentation_mode == "none":
        # No augmentation - resize only
        train_transform_list = [
            LetterBoxTransform(new_shape=(640, 640)),
            T.ToDtype(torch.float32, scale=True)
        ]
    elif augmentation_mode == "geometric":
        # Geometric augmentations only (generic)
        train_transform_list = [
            LetterBoxTransform(new_shape=(640, 640)),
            T.RandomHorizontalFlip(p=0.5),
            T.ToDtype(torch.float32, scale=True)
        ]
    else:  # "full" - custom augmentations for disco/club environment
        train_transform_list = [
            LetterBoxTransform(new_shape=(640, 640)),
            T.RandomHorizontalFlip(p=0.5),
            T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.05),
            T.RandomGrayscale(p=0.1),
            T.GaussianBlur(kernel_size=5, sigma=(0.1, 2.0)),
            T.ToDtype(torch.float32, scale=True)
        ]
    
    train_transforms = YOLOCompose(train_transform_list)
    print(f"Augmentation mode: {augmentation_mode}")
    
    # VALIDATION transforms (no augmentation!)
    val_transforms = YOLOCompose([
        LetterBoxTransform(new_shape=(640, 640)),
        T.ToDtype(torch.float32, scale=True)
    ])

    # Handle dataset size limiting (for Phase 3 experiments)
    train_data_path = params["train_data_path"]
    if "dataset_size" in params and params["dataset_size"] is not None:
        # Read all training image paths
        with open(train_data_path, 'r') as f:
            all_train_paths = [line.strip() for line in f.readlines()]
        
        desired_size = params["dataset_size"]
        if desired_size < len(all_train_paths):
            # Randomly sample a subset (with shuffle for unbiased selection)
            import random
            random.seed(42)  # Fixed seed for reproducibility
            sampled_paths = random.sample(all_train_paths, desired_size)
            
            # Create temporary file with sampled paths
            import tempfile
            temp_file = tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt')
            for path in sampled_paths:
                temp_file.write(path + '\n')
            temp_file.close()
            train_data_path = temp_file.name
            
            print(f"📊 Dataset size limiting: Using {desired_size}/{len(all_train_paths)} training images")
        else:
            print(f"📊 Dataset size: Using all {len(all_train_paths)} training images")

    train_dataset = JägerBombDataset(train_data_path, transforms=train_transforms)
    train_loader = DataLoader(
        train_dataset,
        batch_size=params["batch_size"],
        shuffle=True,
        collate_fn=collate_fn,
        generator=torch.Generator(device)
    )

    val_dataset = JägerBombDataset(params["val_data_path"], transforms=val_transforms)
    val_loader = DataLoader(
        val_dataset,
        batch_size=params["batch_size"],
        shuffle=False,
        collate_fn=collate_fn,
        generator=torch.Generator(device)
    )
    
    # Test dataset (no augmentation!)
    test_dataset = JägerBombDataset(params["test_data_path"], transforms=val_transforms)
    test_loader = DataLoader(
        test_dataset,
        batch_size=params["batch_size"],
        shuffle=False,
        collate_fn=collate_fn,
        generator=torch.Generator(device)
    )
    
    # Setup learning rate scheduler
    lr_scheduler_type = params.get("lr_scheduler", "fixed")  # Default to fixed
    
    if lr_scheduler_type == "cosine":
        def one_cycle_lr(epoch):
            """Cosine learning rate schedule from 1.0 to lrf over epochs."""
            lrf = 0.01  # final learning rate factor (1% of initial)
            # Cosine annealing: starts at 1.0, ends at lrf
            return lrf + (1 - lrf) * 0.5 * (1 + math.cos(math.pi * epoch / params["epochs"]))
        
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=one_cycle_lr)
        print(f"Learning rate scheduler: cosine decay from {lr:.6f} to {lr * 0.01:.6f} over {params['epochs']} epochs")
    else:
        # Fixed learning rate (no scheduler)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda epoch: 1.0)
        print(f"Learning rate: fixed at {lr:.6f}")

    #print("Dataset length:", len(train_dataset))
    #print("Testing __getitem__")
    #print(train_dataset[3])

    # Select loss function based on configuration
    loss_type = params.get("loss_type", "standard")
    if loss_type == "spatial_consistency":
        loss_fn = JägerBombLoss(torch_model,lamda_rate=1)
        print("🎯 Using spatial consistency loss (JägerBombLoss)")
    else:
        loss_fn = v8DetectionLoss(torch_model)
        print("📦 Using standard YOLO loss (v8DetectionLoss)")
    
    cfg = TrainingConfig(
        model=torch_model,
        optimizer=optimizer,
        scheduler=scheduler,
        train_dataloader=train_loader,
        val_dataloader=val_loader,
        test_dataloader=test_loader,
        loss_fn=loss_fn,
        device=device,
        epochs=params["epochs"],
        log_interval=params["log_interval"],
        yolo_model=model,
        use_ema=params.get("use_ema", False),
        freeze_dfl=params.get("freeze_dfl", False),
        experiment_name=experiment_name,
        loss_type=loss_type
    )
    
    save_cfg = SaveConfig(
        save_epoch_interval=params["save_interval"],
        save_path=params["save_path"]
    )
    trainer = JägerBombTrainer(cfg, save_cfg)

    trainer.train()