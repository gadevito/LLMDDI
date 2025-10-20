#
# Use Optuna to find out the best LoRA config to fine-tune the model
#
import argparse
import optuna
from optuna.trial import Trial
import os
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx_lm import load #, generate
#from mlx_lm.utils import generate_step
import numpy as np
from pathlib import Path
import math
import json
import yaml
import types
#import time
from mlx_lm.tuner.trainer import TrainingArgs, evaluate, train
#from mlx.utils import tree_flatten
from mlx_lm.tuner.utils import (print_trainable_parameters, linear_to_lora_layers)
from mlx_lm.tuner.datasets import load_dataset
from mlx_lm.utils import save_config
#from sqlalchemy.exc import SQLAlchemyError

class OptunaCallbackMedian:
    def __init__(self, trial, min_prune_step=20, patience=0):
        self.trial = trial
        self.min_prune_step = min_prune_step
        self.patience = patience
        self.best = float('inf')
        self.no_improve = 0

    def on_train_loss_report(self, info):
        step = info.get("iteration", info.get("step"))
        loss = info.get("train_loss", info.get("loss"))
        if step is None or loss is None:
            return
        self.trial.report(loss, step)
        if step < self.min_prune_step:
            return
        if loss < self.best - 1e-6:
            self.best = loss
            self.no_improve = 0
        else:
            self.no_improve += 1
        if self.no_improve >= self.patience or self.trial.should_prune():
            raise optuna.exceptions.TrialPruned()

    def on_val_loss_report(self, info):
        pass  # disabilitata: niente eval intermedie

class OptunaCallback:
    def __init__(self, trial: Trial, min_prune_step: int = 40, patience: int = 1):
        self.trial = trial
        self.min_prune_step = min_prune_step
        self.best_loss = float('inf')
        self.patience = patience
        self.no_improve_count = 0

    def on_train_loss_report(self, val_info):
        pass

    def on_val_loss_report(self, val_info):
        current_step = val_info["iteration"]
        current_loss = val_info["val_loss"]

        # Report immediatelly to have an early pruning 
        self.trial.report(current_loss, current_step)

        best_attr = self.trial.user_attrs.get("best_val_loss", float("inf"))
        if current_loss < best_attr - 1e-6:
            self.trial.set_user_attr("best_val_loss", float(current_loss))
            self.trial.set_user_attr("best_step", int(current_step))

        if current_step < self.min_prune_step:
            return

        # Patience: prune if it doesn't improve 
        if current_loss < self.best_loss - 1e-6:
            self.best_loss = current_loss
            self.no_improve_count = 0
        else:
            self.no_improve_count += 1

        if self.no_improve_count >= self.patience or self.trial.should_prune():
            raise optuna.exceptions.TrialPruned()
   
def build_schedule(init_lr, decay_steps, end):
    """Create the cosine decay scheduler """
    return optim.cosine_decay(init=init_lr, decay_steps=decay_steps, end=end)


def parse_args():
    parser = argparse.ArgumentParser(description='Hyperparameter optimization for MLX models')
    
    # Base parameters
    parser.add_argument('--model', type=str, default="microsoft/Phi-3.5-mini-instruct",
                      help='Path to the model or model identifier')
    parser.add_argument('--output-dir', type=str, default="optimization_results",
                      help='Directory to save optimization results')
    parser.add_argument('--data', type=str, required=True,
                      help='Path to the dataset')
    parser.add_argument('--max-seq-length', type=int, default=1024, #2048
                      help='Maximum sequence length')
    parser.add_argument('--seed', type=int, default=42,
                      help='Random seed')
    
    # Training params
    parser.add_argument('--iters', type=int, default=30, #150, #300
                      help='Number of training iterations')
    parser.add_argument('--steps-per-report', type=int, default=10,
                      help='Steps between loss reporting')
    parser.add_argument('--steps-per-eval', type=int, default=60, #50
                      help='Steps between evaluations')
    parser.add_argument('--save-every', type=int, default=1000000000, #150
                      help='Steps between saving model')
    parser.add_argument('--val-batches', type=int, default=3,#25
                      help='Number of validation batches')
    parser.add_argument('--save-model', action='store_true',
                      help='Whether to save the best model during training')

    parser.add_argument('--ignore_pruner', action='store_true',
                      help='Whether to ignore pruner')

    parser.add_argument('--pruner_factor', type=int, default=2,#3
                      help='Pruner factor')

    parser.add_argument('--patience', type=int, default=1,# aggressive
                      help='Number of allowed successive non-improved iterations')

    # Optuna config
    parser.add_argument('--optuna-config', type=str, required=True,
                      help='Path to the YAML file containing Optuna configuration')
    
    return parser.parse_args()

def load_optuna_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def suggest_parameters(trial: Trial, param_config):
    params = {}
    for param_name, param_specs in param_config.items():
        ptype = param_specs['type']
        if ptype == 'float':
            low = float(param_specs['low'])
            high = float(param_specs['high'])
            use_log = param_specs.get('log', False)
            step = param_specs.get('step', None)

            if use_log and (low <= 0 or high <= 0):
                raise ValueError(f"Log scale requires positive values for {param_name}")

            if use_log:
                params[param_name] = trial.suggest_float(
                    param_name, low, high, log=True
                )
            elif step is not None:
                params[param_name] = trial.suggest_float(
                    param_name, low, high, step=float(step)
                )
            else:
                params[param_name] = trial.suggest_float(
                    param_name, low, high
                )

        elif ptype == 'int':
            params[param_name] = trial.suggest_int(
                param_name, param_specs['low'], param_specs['high']
            )

        elif ptype == 'categorical':
            params[param_name] = trial.suggest_categorical(
                param_name, param_specs['choices']
            )
    return params

def evaluate_model(model: nn.Module, dataset, tokenizer, batch_size, num_batches, max_seq_length):
    """Evaluate the model using loss and perplexity"""
    model.eval()
    
    val_loss = evaluate(
        model=model,
        dataset=dataset,
        tokenizer=tokenizer,
        batch_size=batch_size,
        num_batches=num_batches,
        max_seq_length=max_seq_length,
    )
    
    val_ppl = math.exp(val_loss)
    
    print(f"Validation loss {val_loss:.3f}, Validation ppl {val_ppl:.3f}")
    
    return val_loss, val_ppl

def train_model(model, tokenizer, args, trial=None):
    """Training as reported in the MLX lora.py script """
    model.freeze()
    
    # Convert linear layers to LoRA layers
    linear_to_lora_layers(
        model,
        args.num_layers,
        args.lora_parameters,
        use_dora=False
    )

    print_trainable_parameters(model)

    print("=================================")
    print(f"LEARNING RATE: {args.learning_rate}")
    print(f"BATCH SIZE: {args.batch_size}")
    print(f"NUM LAYERS: {args.num_layers}")
    print(f"RANK: {args.lora_parameters['rank']}")
    print(f"ALPHA: {args.lora_parameters['alpha']}")
    print(f"DROPOUT: {args.lora_parameters['dropout']}")
    print(f"SCALE: {args.lora_parameters['scale']}")
    print("=================================\n")

    # Create adapter path
    adapter_path = Path(args.output_dir) / "adapters"
    adapter_path.mkdir(parents=True, exist_ok=True)

    adapter_file = adapter_path / "adapters.safetensors"
    
    # Create a serializable configuration version
    config_to_save = {
        'learning_rate': args.learning_rate,
        'lr_end': args.lr_end if hasattr(args, 'lr_end') else 0,
        'batch_size': args.batch_size,
        'num_layers': args.num_layers,
        'max_seq_length': args.max_seq_length,
        'iters': args.iters,
        'steps_per_eval': args.steps_per_eval,
        'save_every': args.save_every,
        'val_batches': args.val_batches,
        'lora_parameters': args.lora_parameters
    }
    
    save_config(config_to_save, adapter_path / "adapter_config.json")

    # init training args
    training_args = TrainingArgs(
        batch_size=args.batch_size,
        iters=args.iters,
        val_batches=args.val_batches,
        steps_per_report=args.steps_per_report,  
        steps_per_eval=args.steps_per_eval,
        steps_per_save=args.save_every,
        adapter_file=adapter_file,
        max_seq_length=args.max_seq_length,
        grad_checkpoint=False
    )

    model.train()
    
    # Create scheduler and optimizer
    lr_schedule = build_schedule(
        init_lr=args.learning_rate,
        decay_steps=args.iters,
        end=args.lr_end if hasattr(args, 'lr_end') else 0
    ) #if hasattr(args, 'lr_end') else args.learning_rate
    
    optimizer = optim.Adam(learning_rate=lr_schedule)

    # Create callback if trial is provided
    training_callback = OptunaCallback(
        trial,
        min_prune_step=args.steps_per_eval*2,  # eg. 5
        patience=args.patience
    ) if trial is not None else None

    # Train model
    train(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        optimizer=optimizer,
        train_dataset=args.train_dataset,
        val_dataset=args.valid_dataset,
        training_callback=training_callback
    )
    
    val_loss, val_ppl = evaluate_model(
        model=model,
        dataset=args.valid_dataset,
        tokenizer=tokenizer,
        batch_size=args.batch_size,
        num_batches=args.val_batches,
        max_seq_length=args.max_seq_length
    )
    
    return model, val_loss, val_ppl


def create_objective(args, optuna_config):
    def objective(trial: Trial):
        # Parameters in the YAML configuration
        params = suggest_parameters(trial, optuna_config['parameter_space'])
        
        try:
            # Load model and tokenizer
            print(f"Loading model from {args.model}")
            model, tokenizer = load(args.model)
            
            # Create args for datasets
            dataset_args = types.SimpleNamespace(
                data=args.data,
                max_seq_length=args.max_seq_length,
                train=True,
                test=False,
                val_batches=args.val_batches,
                test_batches=None,
                batch_size=params['batch_size']
            )
            
            # Load datasets
            train_dataset, valid_dataset, _ = load_dataset(dataset_args, tokenizer)
            
            try:
                lrx = float(params['learning_rate'])
                lre = lrx * 0.1
            except:
                lrx = params['learning_rate']
                lre = params.get('lr_end',0)
            # Configure training args 
            training_args = types.SimpleNamespace(
                learning_rate=lrx, #params['learning_rate'],
                lr_end=lre,
                patience = args.patience,
                pruner_factor = args.pruner_factor,
                ignore_pruner = args.ignore_pruner,
                batch_size=params['batch_size'],
                num_layers=params['num_layers'],
                max_seq_length=args.max_seq_length,
                iters=args.iters,
                steps_per_eval=args.steps_per_eval,
                steps_per_report = args.steps_per_report,
                save_every=args.save_every,
                val_batches=args.val_batches,
                save_model=args.save_model,
                output_dir=args.output_dir,
                train_dataset=train_dataset,
                valid_dataset=valid_dataset,
                lora_parameters={
                    "keys": optuna_config["lora_keys"],
                    "rank": params['lora_rank'],
                    "alpha": params['lora_rank'] * params['lora_alpha_scaling'],
                    "dropout": params['lora_dropout'],
                    "scale": params['lora_scale']
                }
            )
            
            # Training
            model, val_loss, val_ppl = train_model(model, tokenizer, training_args, trial)

            init_par = False
            if init_par:
                # Report validation loss and check pruning
                for step in range(training_args.steps_per_eval, training_args.iters, training_args.steps_per_eval):
                    trial.report(val_loss, step) # report loss at each validation step
                    if trial.should_prune():
                        raise optuna.exceptions.TrialPruned()
                
            # Save the trial results 
            trial_dir = Path(args.output_dir) / f"trial_{trial.number}"
            trial_dir.mkdir(parents=True, exist_ok=True)
            results = {
                "params": params,
                "metrics": {
                    "validation_loss": val_loss,
                    "validation_perplexity": val_ppl
                }
            }
            with open(trial_dir / "results.json", "w") as f:
                json.dump(results, f, indent=4)
            
            return val_loss
            
        except Exception as e:
            print(f"Trial failed: {e}")
            raise  # raise the exception 
            
    return objective


def save_topk_trials(study, k, out_path):
    # prendi solo i trial che hanno best_val_loss registrato
    trials = [t for t in study.trials if "best_val_loss" in t.user_attrs]
    if not trials:
        print("Nessun trial con best_val_loss nei user_attrs. Controlla la callback.")
        return

    trials.sort(key=lambda t: t.user_attrs["best_val_loss"])
    topk = trials[:k]

    payload = []
    for t in topk:
        payload.append({
            "trial_number": t.number,
            "state": str(t.state),  # COMPLETE o PRUNED
            "best_val_loss": t.user_attrs["best_val_loss"],
            "best_step": t.user_attrs.get("best_step"),
            "params": t.params
        })

    import json, os
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"Salvate {len(payload)} configurazioni candidate in {out_path}")

def main():
    args = parse_args()
    
    # Create the output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load the Optuna config
    optuna_config = load_optuna_config(args.optuna_config)
    
    # Se the seed for reproducibility
    np.random.seed(args.seed)
    
    total_trials = optuna_config['optimization']['n_trials']
    startup_trials = 3 #5 

    # Create the SQLite storage
    storage_name = f"sqlite:///{args.output_dir}/optuna.db"
    study_name = optuna_config['optimization']['study_name']

    try:
        # Try to load an existing study
        study = optuna.load_study(
            study_name=study_name,
            storage=storage_name
        )
        print(f"Resumed existing study '{study_name}' with {len(study.trials)} trials")
    except:
        #pruner_median = optuna.pruners.MedianPruner(
        #        n_startup_trials=startup_trials,
        #        n_warmup_steps=20, #50, #20,
        #        interval_steps=20 #50, #20
        #    )
        pruner_to_set = None
        if not args.ignore_pruner:
            pruner_to_set = optuna.pruners.SuccessiveHalvingPruner(
                min_resource=args.steps_per_eval,   # es. 5, allineato alle eval
                reduction_factor=args.pruner_factor,
                min_early_stopping_rate=0
            )
        # It the study doesn't exist, create a new one
        study = optuna.create_study(
            study_name=study_name,
            storage=storage_name,
            direction=optuna_config['optimization']['direction'],
            sampler=optuna.samplers.TPESampler(seed=args.seed),
            pruner=pruner_to_set,
            load_if_exists=True  # This is vital to restart stopped optimizations
        )
        print(f"Created new study '{study_name}'")

    # Count the remaining trials 
    remaining_trials = total_trials - len(study.trials)
    if remaining_trials <= 0:
        print("All trials completed!")
        return

    print(f"Running {remaining_trials} remaining trials...")
    
    # Create the objective using optuna config
    objective = create_objective(args, optuna_config)
    
    # Perform optimization
    study.optimize(
        objective,
        n_trials=remaining_trials
    )
    
    # Print and save the results
    print("\nBest trial:")
    trial = study.best_trial
    print(f"  Value: {trial.value}")
    print("  Params: ")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")
    
    # Save the best results
    results = {
        "best_value": trial.value,
        "best_params": trial.params,
        "study_statistics": {
            "n_trials": len(study.trials),
            "n_completed_trials": len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]),
            "n_failed_trials": len([t for t in study.trials if t.state == optuna.trial.TrialState.FAIL])
        }
    }
    
    with open(os.path.join(args.output_dir, "optimization_results.json"), "w") as f:
        json.dump(results, f, indent=4)

    topk_path = os.path.join(args.output_dir, "topk_candidates.json")
    save_topk_trials(study, k=7, out_path=topk_path)

if __name__ == "__main__":
    main()