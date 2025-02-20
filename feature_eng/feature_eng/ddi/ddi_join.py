#
# Add the LLM processed DDI to the drug dataset
#
import pickle
import csv
import argparse
import copy 
from tqdm import tqdm

def load_pickle(file_path):
    """Load pickle file."""
    with open(file_path, 'r') as file:
        return pickle.load(file)

def save_pickle(data, file_path):
    """Save data to pickle file."""
    with open(file_path, 'wb') as file:
        pickle.dump(data, file)

def load_csv(file_path):
    """Load CSV file into a dictionary for quick lookup."""
    interaction_dict = {}
    with open(file_path, newline='', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            interaction_dict[row['Interaction_Description']] = row
    return interaction_dict

def enrich_drug_data(drug_data, interaction_dict):
    """Enrich drug data with interaction details from CSV."""
    enriched_data = []
    total_ddi = 0
    total_rows = 0
    for drug in tqdm(drug_data, desc="Enriching drug data", unit="drug"):
        total_rows += 1
        if 'drug_interactions' in drug and drug['drug_interactions']:
            for interaction in drug['drug_interactions']:
                total_ddi +=1
                new_entry = drug.copy()  # Copy the original drug data
                new_entry['drugbank_id_ddi'] = interaction.get('drugbank_id', None)
                new_entry['drug_name_ddi'] = interaction.get('name', None)
                new_entry['interaction_description'] = interaction.get('description', None)

                # Lookup description in CSV
                desc = new_entry['interaction_description']
                if desc in interaction_dict:
                    csv_row = interaction_dict[desc]
                    new_entry['interaction_classification'] = csv_row.get('interaction_classification', None)
                    new_entry['pharmacokinetic_effect'] = csv_row.get('pharmacokinetic_effect', None)
                    new_entry['pharmacodynamic_effect'] = csv_row.get('pharmacodynamic_effect', None)
                    new_entry['severity'] = csv_row.get('severity', None)
                    new_entry['timing'] = csv_row.get('timing', None)
                    # Assuming target_bin_ddi needs to be calculated or extracted from somewhere
                    new_entry['target_bin_ddi'] = 1  # True

                enriched_data.append(new_entry)
        else:
            new_entry = drug  # Copy the original drug data
            new_entry['drugbank_id_ddi'] = None
            new_entry['drug_name_ddi'] =  None
            new_entry['interaction_description'] = None
            new_entry['interaction_classification'] = None
            new_entry['pharmacokinetic_effect'] = None
            new_entry['pharmacodynamic_effect'] = None
            new_entry['severity'] = None
            new_entry['timing'] = None
            new_entry['target_bin_ddi'] = 0 # False
            # If no interactions, append the drug as is
            enriched_data.append(new_entry)
    
    print("Total rows", total_rows)
    print("Total ddi", total_ddi)
    return enriched_data

def main(original_pickle, csv_file, output_pickle):
    # Load files
    drugs = load_pickle(original_pickle)
    interaction_dict = load_csv(csv_file)

    # Enrich drug data
    enriched_drug_data = enrich_drug_data(drugs, interaction_dict)    

    # Save enriched data
    save_pickle(enriched_drug_data, output_pickle)
    print(f"Enriched data saved to {output_pickle}")

    print("Total rows", len(enriched_drug_data))
    total_ddi = 0
    for r in enriched_drug_data:
        if r['drugbank_id_ddi']:
            total_ddi +=1
    print("Total ddi", total_ddi)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process drug interaction data.')
    parser.add_argument('--input_pickle', type=str, required=True, help='Path to the input pickle file')
    parser.add_argument('--csv_file', type=str, required=True, help='Path to the CSV file with interaction classifications')
    parser.add_argument('--output_pickle', type=str, required=True, help='Path to the output pickle file')
    
    args = parser.parse_args()
    main(args.input_pickle, args.csv_file, args.output_pickle)