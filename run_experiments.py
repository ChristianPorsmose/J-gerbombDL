"""
Run multiple experiment configurations sequentially.

Usage:
    python run_experiments.py
"""

import yaml
import subprocess
import sys
from pathlib import Path
from datetime import datetime

# Import all experiment configs
from experiment_configs import (
    PHASE1A_SGD_STANDARD,
    PHASE1A_SGD_CONSERVATIVE,
    PHASE1A_ADAMW_STANDARD,
    PHASE1A_ADAMW_CONSERVATIVE,
    PHASE1A_ADAMW_AGGRESSIVE,
)


def run_experiment(config_name: str):
    """
    Run a single experiment by passing config name directly to main.py
    
    Args:
        config_name: Name of the config variable from experiment_configs.py
    
    Returns:
        bool: True if experiment succeeded, False otherwise
    """
    print(f"\n{'='*80}")
    print(f"🚀 Starting experiment: {config_name}")
    print(f"{'='*80}\n")
    
    # Run training with config dict directly (no file I/O)
    try:
        result = subprocess.run(
            ['python', 'main.py', '--config-dict', config_name],
            check=True,
            capture_output=False,  # Show output in real-time
            text=True
        )
        print(f"\n✅ Experiment '{config_name}' completed successfully!\n")
        return True
    
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Experiment '{config_name}' failed with error code {e.returncode}\n")
        return False
    
    except KeyboardInterrupt:
        print(f"\n⚠️  Experiment '{config_name}' interrupted by user\n")
        return False


def run_experiment_queue(config_names: list[str], continue_on_error: bool = True):
    """
    Run a queue of experiments sequentially.
    
    Args:
        config_names: List of config variable names from experiment_configs.py
        continue_on_error: If True, continue to next experiment even if one fails
    
    Returns:
        dict: Summary of experiment results
    """
    start_time = datetime.now()
    
    results = {
        'total': len(config_names),
        'succeeded': 0,
        'failed': 0,
        'experiments': []
    }
    
    print(f"\n{'#'*80}")
    print(f"🧪 EXPERIMENT QUEUE")
    print(f"{'#'*80}")
    print(f"Total experiments: {len(config_names)}")
    print(f"Continue on error: {continue_on_error}")
    print(f"Start time: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'#'*80}\n")
    
    for i, config_name in enumerate(config_names, 1):
        print(f"\n[{i}/{len(config_names)}] Processing: {config_name}")
        
        success = run_experiment(config_name)
        
        results['experiments'].append({
            'name': config_name,
            'success': success
        })
        
        if success:
            results['succeeded'] += 1
        else:
            results['failed'] += 1
            if not continue_on_error:
                print(f"\n⚠️  Stopping queue due to failure (continue_on_error=False)")
                break
    
    # Print summary
    end_time = datetime.now()
    duration = end_time - start_time
    
    print(f"\n{'#'*80}")
    print(f"📊 EXPERIMENT QUEUE SUMMARY")
    print(f"{'#'*80}")
    print(f"Total experiments: {results['total']}")
    print(f"✅ Succeeded: {results['succeeded']}")
    print(f"❌ Failed: {results['failed']}")
    print(f"⏱️  Duration: {duration}")
    print(f"End time: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"\nResults by experiment:")
    for exp in results['experiments']:
        status = "✅" if exp['success'] else "❌"
        print(f"  {status} {exp['name']}")
    print(f"{'#'*80}\n")
    
    return results


if __name__ == "__main__":
    # ========== CONFIGURE YOUR EXPERIMENT QUEUE HERE ==========
    
    # Example: Run all Phase 1A experiments
    EXPERIMENT_QUEUE = [
        "PHASE1A_SGD_CONSERVATIVE",
        "PHASE1A_ADAMW_STANDARD",
        "PHASE1A_ADAMW_CONSERVATIVE",
        "PHASE1A_ADAMW_AGGRESSIVE",
    ]
    
    # Or run a custom subset:
    # EXPERIMENT_QUEUE = [
    #     "PHASE1A_ADAMW_STANDARD",
    #     "PHASE1A_ADAMW_CONSERVATIVE",
    # ]
    
    # ==========================================================
    
    # Run the queue
    results = run_experiment_queue(
        config_names=EXPERIMENT_QUEUE,
        continue_on_error=True  # Set to False to stop on first failure
    )
    
    # Exit with error code if any experiments failed
    if results['failed'] > 0:
        sys.exit(1)
    else:
        sys.exit(0)
