import yaml
import torch
from torch.optim import Adam
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from ultralytics.models import YOLO
import os

from jäger_bomb_dataset import JägerBombDataset
from jäger_bomb_loss import JägerBombLoss
from jäger_bomb_trainer import JägerBombTrainer
from configs import TrainingConfig, SaveConfig
from torchvision.transforms import v2 as T
from types import SimpleNamespace

class LetterBoxTransform:
    def __init__(self, new_shape=(768, 1024), color=(114, 114, 114)):
        self.new_shape = new_shape
        self.color = color
    
    def __call__(self, img):
        # img is a tensor [C, H, W]
        shape = img.shape[1:]  # current shape [H, W]
        new_shape = self.new_shape
        
        # Scale ratio (new / old)
        r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
        
        # Compute padding
        new_unpad = int(round(shape[0] * r)), int(round(shape[1] * r))
        dh, dw = new_shape[0] - new_unpad[0], new_shape[1] - new_unpad[1]  # wh padding
        
        dh /= 2  # divide padding into 2 sides
        dw /= 2
        
        if shape != new_unpad:  # resize
            img = T.Resize(new_unpad)(img)
        
        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        
        # Add padding
        img = T.Pad([left, top, right, bottom], fill=self.color[0])(img)  # Use first color value for all channels
        
        return img


def collate_fn(batch):
    images, targets = zip(*batch)
    padded_targets = pad_sequence(targets, batch_first=True, padding_value=-1)
    images = torch.stack(images, 0)
    return images, padded_targets


def defrost_layers(torch_model):
    for child in torch_model.children():
        for param in child.parameters():
            param.requires_grad = True

if __name__ == "__main__":

    with open("setup.yaml", "r") as f:
        params = yaml.safe_load(f)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.set_default_device(device)

    model = YOLO( params["model"], task="detect").load('yolo11n.pt')
    
    #model.names = {1:"cup", 0:"shot"}
    torch_model = model.model
    torch_model.to(device)

    defrost_layers(torch_model)

    # Convert args dict to SimpleNamespace so it has attributes instead of dict keys
    if isinstance(torch_model.args, dict):
        torch_model.args = SimpleNamespace(**torch_model.args)

    # Now add the hyperparameters as attributes
    torch_model.args = SimpleNamespace(box=15, cls=0.5, dfl=2.25)
    
    optimizer = Adam(model.parameters(), lr=params["lr"])

    # transforms = T.Compose([
    #     LetterBoxTransform(new_shape=(640, 640)),
    #     T.ToDtype(torch.float32,scale=True)
    # ])

    # TRAINING transforms with augmentation
    train_transforms = T.Compose([
        LetterBoxTransform(new_shape=(768, 1024)),
        T.RandomHorizontalFlip(p=0.5),
        T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.05),
        T.RandomGrayscale(p=0.1),
        T.GaussianBlur(kernel_size=5, sigma=(0.1, 2.0)),
        T.ToDtype(torch.float32, scale=True)
    ])
    
    # VALIDATION transforms (no augmentation!)
    val_transforms = T.Compose([
        LetterBoxTransform(new_shape=(768, 1024)),
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

    #print("Dataset length:", len(train_dataset))
    #print("Testing __getitem__")
    #print(train_dataset[3])

    cfg = TrainingConfig(
        model=torch_model,
        optimizer=optimizer,
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