# Jägerbomb glass detector
This contains the code that builds and analyse models to predict jägerbombglass

## Code structure
```
├── analysis
│   ├── __init__.py
│   ├── phase_analysis.py
│   ├── plot_train_val_compare_combined.py
│   └── plotting.py
├── configs.py
├── dataset_final_boxes_yolo/*
├── dataset
│   ├── __init__.py
│   ├── jäger_bomb_dataset.py
│   ├── letter_box_transform.py
│   └── yolo_compose.py
├── engine
│   ├── __init__.py
│   ├── bomb_visualize.py
│   ├── data.py
│   ├── jäger_bomb_loss.py
│   ├── jäger_bomb_trainer.py
│   └── log_helpers.py
├── experiments_scripts
│   ├── __init__.py
│   ├── experiment_configs.py
│   └── run_grid_search.py
├── factory
│   ├── __init__.py
│   └── training_factory.py
├── freezer
│   ├── __init__.py
│   └── freezer.py
├── main.py
├── main.yaml
├── metrics
│   ├── __init__.py
│   ├── jäger_bomb_metric_logger.py
│   ├── jäger_bomb_metric_tracker.py
│   ├── metric_visualization.py
│   └── metrics.py
├── models
│   ├── Adrian.yaml
│   ├── basic_detection.yaml
│   └── custom_yolo.yaml
├── ultralytics/*
├── README.md
├── requirements.txt
├── template.yaml
├── test_model.py
├── utils
│   ├── echo.py
│   ├── experiment_log.py
│   └── utils.py
├── visualize_layers.ipynb
└── yolo11n.pt
```



### Installation

```
pip install -r requirements.txt
```


## Running the Code

```
python main.py --config <config_file>.yaml
```

### Arguments
| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--config` | yaml | `main.yaml` | Model training configuration ( see template.yaml) |

## Running Experiments

```
python -m experiments.run_grid_search "arguments"
```

### Arguments
| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--phase` | phase1, phase2, phase3, single | `single` | Select the experiment phase. `single` requires `--base` and `--grid`. |
| `--base` | Choice from `BASE_CONFIGS` | None | Base configuration for `--phase=single`. Required if phase is `single`. Look in the file to see available |
| `--grid` | learning, custom-loss | None | Which parameter grid to run. Only used with `--phase=single`. |
| `--continue-on-error` | Flag | True | Continue running other configurations even if one fails. |


## Analysis

