#!/bin/bash

# Check if all the parameters have been passed
if [ "$#" -lt 2 ]; then
  echo "Usage: $0 <model name> <epochs>"
  echo "Options for <model name>: deepseek-r1, phi3.5-2b, gemma-2-9b, qwen2.5-3b"
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
  "phi3.5-2b")
    HF_MODEL_NAME="microsoft/Phi-3.5-mini-instruct"
    MODEL_PATH="phi3.5-2b-ft"
    ;;
  "gemma-2-9b")
    HF_MODEL_NAME="mlx-community/gemma-2-9b-it-fp16"
    MODEL_PATH="gemma-2-9b-ft"
    ;;
  "qwen2.5-3b")
    HF_MODEL_NAME="mlx-community/Qwen2.5-3B-Instruct-4bit"
    MODEL_PATH="qwen2.5-3b-ft"
    ;;
  "deepseek-r1")
    HF_MODEL_NAME="mlx-community/deepseek-r1-distill-qwen-1.5b"
    MODEL_PATH="deepseek-r1"
    ;;
  *)
    echo "Option <model name> is not valid: $MODEL_NAME"
    echo "Valid options: deepseek-r1, phi3.5-2b, gemma-2-9b, qwen2.5-3b"
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

# Copy tokenizer.model if the model is gemma-2-9b
if [ "$MODEL_NAME" == "gemma-2-9b" ]; then
  echo "Copying tokenizer.model for gemma-2-9b..."
  cp patch/gemma-2-9b/tokenizer.model "./models/${MODEL_PATH}_${EPOCHS}e/"
fi

# Copy tokenizer.model if the model is mistral-nemo-12b
if [ "$MODEL_NAME" == "deepseek-r1" ]; then
  echo "Remember to copy $MODEL_NAME to an LM Studio folder"
  exit 1
fi

echo
echo "Creating GGUF file...."
# Create the GGUF file
cd llama.cpp
source .venv/bin/activate

python convert_hf_to_gguf.py "../models/${MODEL_PATH}_${EPOCHS}e" --outfile "../models/${MODEL_PATH}_${EPOCHS}e.gguf" --outtype q8_0
echo
echo "Model created: $MODEL_NAME using $EPOCHS epochs"
