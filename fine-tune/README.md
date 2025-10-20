## FINE-TUNING LLMs ###
This folder contains a few scripts:
- setup_env.sh, setup the environment and install libraries
- best_config.py, use Optuna to find out the best parameters for LoRA fine-tuning
- fine_tune.sh, perform LLM fine-tuning
- fine_tune_2k.sh, perform PHI-3.5 fine-tuning on 2k samples
- fine_tune_ablation.sh, perform PHI 3.5 fine-tuning using ony SMILES