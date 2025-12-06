# Jägerbomb glass detector
This contains the code that builds and analyse models to predict jägerbombglass

## folder structure
```
| Folder                 | Description                                             |
|------------------------|---------------------------------------------------------|
| `analysis`             | Scripts for analyzing results and generating plots      |
| `dataset`              | Dataset classes, preprocessing, and data transforms     |
| `engine`               | Core training, evaluation, and loss function            |
| `experiments_scripts`  | Scripts for running experiments and grid searches       |
| `factory`              | Factory modules for setting up training configurations  |
| `freezer`              | for freezing models or layers                           |
| `metrics`              | Metric calculation, tracking, and visualization         |
| `models`               | Model configuration files (YAML)                        |
| `utils`                | Helper functions and utility scripts                    |
```


### Installation

```
pip install -r requirements.txt
```

## Running main training

```
python main.py --config <config_file>.yaml
```

### Arguments
| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--config` | yaml | `main.yaml` | Model training configuration ( see template.yaml) |

## Experiments

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

```
python -m analysis.phase_analysis "arguments"
```

### Arguments
| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--filter` | str| None | filter experiments by name given substring fiter in experiments_results |
| `--grid` | flag | False | plot grid specific plots |
| `-o` `--output` | path | analysis_dir | output folder |