import json
from pathlib import Path
from utils import convert_to_serializable


class JägerBombLogger:

    @staticmethod
    def log_config_params(save_dir, cfg, config_params=None):
        """Log all configuration parameters to a file."""
        config_path = Path(save_dir) / "config.json"
        
        # Convert all values to JSON-serializable format
        
        # Collect all TrainingConfig parameters
        config_data = {}
        
        # Add config_params if available (original YAML/dict config)
        if config_params is not None:
            config_data['original_config'] = convert_to_serializable(config_params)
        
        # Add all TrainingConfig attributes
        config_data['training_config'] = {
            'device': cfg.device,
            'epochs': cfg.epochs,
            'log_interval': cfg.log_interval,
            'use_ema': cfg.use_ema,
            'freeze_dfl': cfg.freeze_dfl,
            'experiment_name': cfg.experiment_name,
            'loss_type': cfg.loss_type,
        }
        
        # Add optimizer info
        config_data['optimizer'] = {
            'type': type(cfg.optimizer).__name__,
            'param_groups': []
        }
        for i, pg in enumerate(cfg.optimizer.param_groups):
            pg_info = {
                'group_id': i,
                'num_params': len(pg['params']),
                'lr': pg.get('lr', 'N/A'),
                'weight_decay': pg.get('weight_decay', 'N/A'),
                'momentum': pg.get('momentum', 'N/A'),
                'betas': pg.get('betas', 'N/A'),
            }
            config_data['optimizer']['param_groups'].append(pg_info)
        
        # Add scheduler info
        config_data['scheduler'] = {
            'type': type(cfg.scheduler).__name__,
        }
        
        # Add dataloader info
        config_data['dataloaders'] = {
            'train': {
                'batch_size': cfg.train_dataloader.batch_size,
                'num_batches': len(cfg.train_dataloader),
                'dataset_size': len(cfg.train_dataloader.dataset),
            },
            'val': {
                'batch_size': cfg.val_dataloader.batch_size,
                'num_batches': len(cfg.val_dataloader),
                'dataset_size': len(cfg.val_dataloader.dataset),
            },
            'test': {
                'batch_size': cfg.test_dataloader.batch_size,
                'num_batches': len(cfg.test_dataloader),
                'dataset_size': len(cfg.test_dataloader.dataset),
            }
        }
        
        # Add loss function info
        config_data['loss_function'] = {
            'type': type(cfg.loss_fn).__name__,
        }
        
        # Add model info
        config_data['model'] = {
            'type': type(cfg.model).__name__,
            'total_params': sum(p.numel() for p in cfg.model.parameters()),
            'trainable_params': sum(p.numel() for p in cfg.model.parameters() if p.requires_grad),
        }
        
        with open(config_path, 'w') as f:
            json.dump(config_data, f, indent=2)
        
        print(f"📝 Configuration parameters saved → {config_path}")
        
        # Also print key parameters to console
        print("\n" + "="*60)
        print("🔧 TRAINING CONFIGURATION")
        print("="*60)
        print(f"  Experiment: {config_data['training_config']['experiment_name']}")
        print(f"  Device: {config_data['training_config']['device']}")
        print(f"  Epochs: {config_data['training_config']['epochs']}")
        print(f"  Optimizer: {config_data['optimizer']['type']}")
        print(f"  Loss Type: {config_data['training_config']['loss_type']}")
        print(f"  Use EMA: {config_data['training_config']['use_ema']}")
        print(f"  Freeze DFL: {config_data['training_config']['freeze_dfl']}")
        print(f"  Train Dataset: {config_data['dataloaders']['train']['dataset_size']} samples")
        print(f"  Val Dataset: {config_data['dataloaders']['val']['dataset_size']} samples")
        print(f"  Test Dataset: {config_data['dataloaders']['test']['dataset_size']} samples")
        print(f"  Model Params: {config_data['model']['trainable_params']:,} trainable / {config_data['model']['total_params']:,} total")
        if 'original_config' in config_data:
            print("\n  Original Config Parameters:")
            for key, value in config_data['original_config'].items():
                if key not in ['train_data_path', 'val_data_path', 'test_data_path', 'model']:
                    print(f"    {key}: {value}")
        print("="*60 + "\n")