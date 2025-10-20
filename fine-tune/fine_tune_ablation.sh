#!/bin/bash

# Check if all the parameters have been passed
if [ "$#" -lt 2 ]; then
  echo "Usage: $0 <model name> <epochs>"
  echo "Options for <model name>: phi3.5-2b-a4, phi3.5-2b-a1, phi3.5-2b-a2, phi3.5-2b-a3, phi3.5-2b-a5, phi3.5-2b-a6"
  echo "<epochs> must be a positive integer: 1, 3, or 9"
  echo "[mode] is optional and can be 'lora', 'dora', or 'full'; default is 'lora'"
  exit 1
fi

MODEL_NAME=$1
EPOCHS=$2
MODE=${3:-lora} # Default value for MODE is 'lora'
HF_MODEL_NAME=""
MODEL_PATH=""

# Check if the number of epochs is valid
if ! [[ "$EPOCHS" =~ ^(1|3|5|9)$ ]]; then
  echo "Error: <epochs> must be 1, 3, 5, or 9."
  exit 1
fi

# Check if MODE is valid
if ! [[ "$MODE" =~ ^(lora|dora|full)$ ]]; then
  echo "Error: [mode] must be 'lora', 'dora', or 'full'."
  exit 1
fi

# Determine values for hf model name and model path based on the model_name
case $MODEL_NAME in
  "phi3.5-2b-a1")
    HF_MODEL_NAME="microsoft/Phi-3.5-mini-instruct"
    MODEL_PATH="phi3.5-2b-a1-ft"
    ;;
  "phi3.5-2b-a2")
    HF_MODEL_NAME="microsoft/Phi-3.5-mini-instruct"
    MODEL_PATH="phi3.5-2b-a2-ft"
    ;;
  "phi3.5-2b-a3")
    HF_MODEL_NAME="microsoft/Phi-3.5-mini-instruct"
    MODEL_PATH="phi3.5-2b-a3-ft"
    ;;
  "phi3.5-2b-a4")
    HF_MODEL_NAME="microsoft/Phi-3.5-mini-instruct"
    MODEL_PATH="phi3.5-2b-a4-ft"
    ;;
  "phi3.5-2b-a5")
    HF_MODEL_NAME="microsoft/Phi-3.5-mini-instruct"
    MODEL_PATH="phi3.5-2b-a5-ft"
    ;;
  "phi3.5-2b-a6")
    HF_MODEL_NAME="microsoft/Phi-3.5-mini-instruct"
    MODEL_PATH="phi3.5-2b-a6-ft"
    ;;
  *)
    echo "Option <model name> is not valid: $MODEL_NAME"
    echo "Valid options: phi3.5-2b-a4, phi3.5-2b-a1, phi3.5-2b-a2, phi3.5-2b-a3, phi3.5-2b-a5, phi3.5-2b-a6"
    exit 1
    ;;
esac

# Adjust configuration filename based on MODE
if [ "$MODE" != "lora" ]; then
  CONFIG_SUFFIX="_${MODE}"
  echo "${MODE}"
else
  CONFIG_SUFFIX=""
fi

source .venv/bin/activate

# Login to Hugging Face
huggingface-cli login

# Download the model
huggingface-cli download "$HF_MODEL_NAME"

# Fine-tuning using mlx_lm
python -m mlx_lm.lora --config "${MODEL_NAME}_config_${EPOCHS}e${CONFIG_SUFFIX}.yaml" --fine-tune-type "${MODE}"
echo
echo "Starting fusion..."
# Adapters fusion
python -m mlx_lm.fuse --model "$HF_MODEL_NAME" --adapter-path "adapters/adapters_${MODEL_NAME}_${EPOCHS}e" --save-path "models/${MODEL_PATH}_${EPOCHS}e" --de-quantize

echo
echo "Creating GGUF file...."
# Create the GGUF file
cd llama.cpp
source .venv/bin/activate

python convert_hf_to_gguf.py "../models/${MODEL_PATH}_${EPOCHS}e" --outfile "../models/${MODEL_PATH}_${EPOCHS}e.gguf" --outtype q8_0
echo
echo "Model created: $MODEL_NAME using $EPOCHS epochs"
