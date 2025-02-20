#!/bin/bash

# Create the Python virtual environment
python3 -m venv .venv

# Activate the environment
source .venv/bin/activate

# Install mlx and huggingface libraries
pip install -U mlx-lm
pip install huggingface_hub
pip install "huggingface_hub[cli]"
pip install optuna

git clone https://github.com/ggerganov/llama.cpp.git

cd llama.cpp||exit
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
exit

echo "Setup completed."