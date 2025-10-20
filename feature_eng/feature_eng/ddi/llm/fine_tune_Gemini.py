#
# The following script opens the the dataset, and uses gpt-4o to classify the ddi
# 

import argparse
import json
import google.generativeai as genaix
import os


def main(training, model, ephocs, batch_size):
    training_dataset = []
    
    # Read the JSONL file
    with open(training, 'r', encoding='utf-8') as jsonl_file:
        for line in jsonl_file:
            data = json.loads(line.strip())
            
            # Extract conversations
            conversations = data.get('conversations', [])
            
            system_content = ""
            human_content = ""
            gpt_content = ""
            
            # Extract contents based on roles
            for conv in conversations:
                if conv['role'] == 'system':
                    system_content = conv['content']
                elif conv['role'] == 'human':
                    human_content = conv['content']
                elif conv['role'] == 'gpt':
                    gpt_content = conv['content']
            
            # Join system and human content for input
            input_text = system_content + "\n\n" + human_content
            
            # Add [input, output] to the dataset
            training_dataset.append({"text_input":input_text, "output":gpt_content})
    

    genaix.configure(api_key=os.environ.get("GOOGLE_API_KEY"))

    print("LIST OF TRAINABLE MODELS====>\n")
    for m in genaix.list_models():
        if "createTunedModel" in m.supported_generation_methods:
            print(m.name)
    print("================\n")

    operation = genaix.create_tuned_model(
        # You can use a tuned model here too. Set `source_model="tunedModels/..."`
        source_model=f"models/{model}",
        training_data=training_dataset,
        id = "geminiddi",
        epoch_count = ephocs,
        batch_size=batch_size,
        learning_rate=0.001,
    )

    import time

    for status in operation.wait_bar():
        time.sleep(30)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine tune a Gemini model.")
    parser.add_argument('jsonl_training_in', type=str, help='JSONL file for the training set.')

    parser.add_argument('--model', type=str, default='gemini-1.5-pro', help='Model to use for the fine-tuning.')
    parser.add_argument('--ephocs', type=int, default=3, help='Number of ephocs.')
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size.')

    args = parser.parse_args()

    main(args.jsonl_training_in, args.model, args.ephocs, args.batch_size)

