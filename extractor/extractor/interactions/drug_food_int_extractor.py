import pickle
import pandas as pd
from tqdm import tqdm
import argparse

#
# Extract distinct drug-drug interaction descriptions 
#
def extract_unique_interaction_descriptions(pickle_file, output_csv):
    # Load the pickle file
    with open(pickle_file, 'rb') as f:
        drugs_data = pickle.load(f)
    
    # Initialize a set to track interaction descriptions
    unique_descriptions = set()

    # Iterate drugs 
    for drug in tqdm(drugs_data, desc="Processing Drugs", unit="drug"):
        # Check if there are drug-drug interactions
        if 'food_interactions' in drug:
            for interaction in drug['food_interactions']:
                if interaction:
                    unique_descriptions.add(interaction.strip())

    print(len(unique_descriptions))
    # Convert set to list for creating a new DataFrame 
    descriptions_list = list(unique_descriptions)
    descriptions_df = pd.DataFrame(descriptions_list, columns=['Interaction_Description'])

    # Save the DataFrame to a CSV file f
    descriptions_df.to_csv(output_csv, index=False, encoding='utf-8')

    print(f"Descriptions saved to {output_csv}")

if __name__ == "__main__":
    # Configura l'analizzatore di argomenti
    parser = argparse.ArgumentParser(description='Extract drug-food interaction descriptions.')
    parser.add_argument('--pickle_file', type=str, required=True, help='Drugs pickle file.')
    parser.add_argument('--output_csv', type=str, required=True, help='CSV file name for the output.')
    parser.add_argument('--debug', action='store_true',
                        help='Enable debug logging')
    
    args = parser.parse_args()

    extract_unique_interaction_descriptions(args.pickle_file, args.output_csv)