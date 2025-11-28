import yaml
import torch
from torch.optim import Adam, SGD
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from ultralytics.models import YOLO
import os
import math

from jäger_bomb_dataset import JägerBombDataset
from jäger_bomb_loss import JägerBombLoss
from jäger_bomb_trainer import JägerBombTrainer
from configs import TrainingConfig, SaveConfig
from torchvision.transforms import v2 as T
from types import SimpleNamespace

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
    for param in torch_model.parameters():
        param.requires_grad = True
    print("✅ All model layers unfrozen and ready for training")

if __name__ == "__main__":

    with open("setup.yaml", "r") as f:
        params = yaml.safe_load(f)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.set_default_device(device)

    model = YOLO( params["model"], task="detect").load('yolo11n.pt')
    
    #model.names = {1:"cup", 0:"shot"}
    torch_model = model.model
    torch_model.to(device)

    unfreeze_all_layers(torch_model)
    
    # Convert args dict to SimpleNamespace so it has attributes instead of dict keys
    if isinstance(torch_model.args, dict):
        torch_model.args = SimpleNamespace(**torch_model.args)

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
    lr = 0.001667
    momentum = 0.9
    weight_decay = 0.0005625
    optimizer = torch.optim.AdamW(g[2], lr=lr, betas=(momentum, 0.999), weight_decay=0.0)
    optimizer.add_param_group({"params": g[0], "weight_decay": weight_decay})  # weights with decay
    optimizer.add_param_group({"params": g[1], "weight_decay": 0.0})  # batch norm without decay
    
    print(f"optimizer: AdamW(lr={lr}, momentum={momentum}) with parameter groups {len(g[1])} weight(decay=0.0), {len(g[0])} weight(decay={weight_decay}), {len(g[2])} bias(decay=0.0)")

    # transforms = T.Compose([
    #     LetterBoxTransform(new_shape=(640, 640)),
    #     T.ToDtype(torch.float32,scale=True)
    # ])

    # TRAINING transforms with augmentation
    train_transforms = YOLOCompose([
        LetterBoxTransform(new_shape=(640, 640)),
        T.RandomHorizontalFlip(p=0.5),
        T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.05),
        T.RandomGrayscale(p=0.1),
        T.GaussianBlur(kernel_size=5, sigma=(0.1, 2.0)),
        T.ToDtype(torch.float32, scale=True)
    ])
    
    # VALIDATION transforms (no augmentation!)
    val_transforms = YOLOCompose([
        LetterBoxTransform(new_shape=(640, 640)),
        T.ToDtype(torch.float32, scale=True)
    ])

    train_dataset = JägerBombDataset(params["train_data_path"], transforms=train_transforms)
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
    
    # Setup learning rate scheduler (cosine decay like Ultralytics)
    def one_cycle_lr(epoch):
        """Cosine learning rate schedule from 1.0 to lrf over epochs."""
        lrf = 0.01  # final learning rate factor (1% of initial)
        # Cosine annealing: starts at 1.0, ends at lrf
        return lrf + (1 - lrf) * 0.5 * (1 + math.cos(math.pi * epoch / params["epochs"]))
    
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=one_cycle_lr)
    print(f"Learning rate scheduler: cosine decay from {lr:.6f} to {lr * 0.01:.6f} over {params['epochs']} epochs")

    #print("Dataset length:", len(train_dataset))
    #print("Testing __getitem__")
    #print(train_dataset[3])

    cfg = TrainingConfig(
        model=torch_model,
        optimizer=optimizer,
        scheduler=scheduler,
        train_dataloader=train_loader,
        val_dataloader=val_loader,
        loss_fn=JägerBombLoss(torch_model),
        device=device,
        epochs=params["epochs"],
        log_interval=params["log_interval"],
        yolo_model=model
    )
    
    save_cfg = SaveConfig(
        save_epoch_interval=params["save_interval"],
        save_path=params["save_path"]
    )
    trainer = JägerBombTrainer(cfg, save_cfg)

    trainer.train()