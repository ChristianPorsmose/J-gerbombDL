import copy

LEARNING_RATES = [0.0001, 0.001, 0.01]
MOMENTUMS = [0.9, 0.95, 0.99]
WEIGHT_DECAYS = [0.001, 0.01, 0.1]
PRETRAINED_OPTIONS = [True]
LAMBDA_RATES = [40, 80, 160, 320, 640, 1280]
DATASET_SIZES = [0.5,0.25,0.125]

BASE = {
    "experiment_name": "adamW",
    "model": {
        "type": "models/custom_yolo.yaml",
        "pretrained": True
    },
    "paths": {
        "train": "dataset_final_boxes_yolo/train.txt",
        "val": "dataset_final_boxes_yolo/val.txt",
        "test": "dataset_final_boxes_yolo/test.txt"
    },
    "training": {
        "batch_size": 8,
        "epochs": 100,
        "log_interval": 10,
        "save_interval": 30,
        "save_path": "weights"
    },
    "optimizer": {
        "type": "AdamW",
        "lr": 0.001,
        "momentum": 0.95,
        "weight_decay": 0.001,
        "lr_scheduler": "fixed"
    },
    "freeze": {
        "backbone_layers": 0,
        "dfl": False
    },
    "augmentation": "none",
    "dataset_size": 1,
    "loss": {
        "type": "default",
        "lambda_rate": None
    },
    "use_ema": False
}

# START OPTIMIZER 

OPTIMIZER_PRETRAINED = copy.deepcopy(BASE)
OPTIMIZER_PRETRAINED["experiment_name"] = "optimizer_config_pretrained"
OPTIMIZER_PRETRAINED["freeze"]["backbone_layers"] = 11
OPTIMIZER_PRETRAINED["augmentation"] = "final"


OPTIMIZER_SMALL = copy.deepcopy(BASE)
OPTIMIZER_SMALL["experiment_name"] = "optimizer_config_small"
OPTIMIZER_SMALL["model"]["type"] = "models/Adrian.yaml"
OPTIMIZER_SMALL["model"]["pretrained"] = False
OPTIMIZER_SMALL["freeze"]["backbone_layers"] = 0
OPTIMIZER_SMALL["augmentation"] = "final"

# END OPTIMIZER

# START CUSTOM LOSS FUNCTIONS

CUSTOM_LOSS_BASE = copy.deepcopy(BASE)
CUSTOM_LOSS_BASE["experiment_name"] = "spatial_loss"
CUSTOM_LOSS_BASE["model"]["pretrained"] = False
CUSTOM_LOSS_BASE["loss"]["type"] = "spatial"
CUSTOM_LOSS_BASE["loss"]["lambda_rate"] = LAMBDA_RATES[0]
CUSTOM_LOSS_BASE["augmentation"] = "final"

#TAKE LR, WEIGHT DECAY, MOMENTUM FROM BEST BEFORE
SMALL_CUSTOM_LOSS_BASE = copy.deepcopy(CUSTOM_LOSS_BASE)
SMALL_CUSTOM_LOSS_BASE["experiment_name"] = "jagerloss_no_prop_config_small"
SMALL_CUSTOM_LOSS_BASE["model"]["type"] = "models/Adrian.yaml"
SMALL_CUSTOM_LOSS_BASE["model"]["pretrained"] = False
SMALL_CUSTOM_LOSS_BASE["optimizer"].update({
    "lr": 0.01,
    "momentum": 0.95,
    "weight_decay": 0.01
})

#TAKE BEST FROM SMALL AND APPLY TO PRETRAINED
PRETRAINED_CUSTOM_LOSS_BASE = copy.deepcopy(SMALL_CUSTOM_LOSS_BASE)
PRETRAINED_CUSTOM_LOSS_BASE["experiment_name"] = "jagerloss_no_prop_config_pretrained"
PRETRAINED_CUSTOM_LOSS_BASE["model"]["type"] = "models/custom_yolo.yaml"
PRETRAINED_CUSTOM_LOSS_BASE["model"]["pretrained"] = True
PRETRAINED_CUSTOM_LOSS_BASE["freeze"]["backbone_layers"] = 11
PRETRAINED_CUSTOM_LOSS_BASE["optimizer"].update({
    "lr": 0.01,
    "momentum": 0.99,
    "weight_decay": 0.001
})

# END CUSTOM LOSS FUNCTIONS



# START DATA TEST CONFIGS

SMALL_DATASET_BASE = copy.deepcopy(SMALL_CUSTOM_LOSS_BASE)
SMALL_DATASET_BASE["experiment_name"] = "dataset_size_small"
SMALL_DATASET_BASE["training"]["batch_size"] = 8  # keep same as base
SMALL_DATASET_BASE["loss"]["type"] = "default"
SMALL_DATASET_BASE["optimizer"].update({
    "lr": 0.01,
    "momentum": 0.95,
    "weight_decay": 0.01
})
SMALL_DATASET_BASE["dataset_size"] = 1


PRETRAINED_DATASET_BASE = copy.deepcopy(PRETRAINED_CUSTOM_LOSS_BASE)
PRETRAINED_DATASET_BASE["experiment_name"] = "dataset_size_pretrained"
PRETRAINED_DATASET_BASE["loss"]["type"] = "default"
PRETRAINED_DATASET_BASE["optimizer"].update({
    "lr": 0.01,
    "momentum": 0.99,
    "weight_decay": 0.001
})
PRETRAINED_DATASET_BASE["dataset_size"] = 1  # will be overridden in grid

# END DATA TEST CONFIGS