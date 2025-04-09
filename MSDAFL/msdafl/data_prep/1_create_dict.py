#
# Create the dictionary file for MSDAFL, that maps drugbank ids with integers.
# To this aim, we load all the datasets (train, validation, external datasets) to collect and map all ids.
#

import json
import argparse
import pickle
import os
from rdkit import Chem

def isValidSmiles(smiles):
    try:
        mol = Chem.MolFromSmiles(smiles)
        return mol is not None
    except Exception as e:
        return False
    
def loadPickle(input_pickle):
    with open(input_pickle, 'rb') as f:
        r = pickle.load(f)
    return r



def create_drug_mapping(dataset_path, output_json, non_empty_smiles):
    """
    Create a dictionary to map drug codes
    """
    dataset = [] 
    for filename in os.listdir(dataset_path):
        file_path = os.path.join(dataset_path, filename)
        if os.path.isfile(file_path):
            if file_path.find(".DS_Store") !=-1:
                continue
            print(f"Load {file_path}")
            p = loadPickle(file_path)
            dataset.extend(p)

    # Collect all drug codes
    all_drug_ids = set()
    for item in dataset:
        if not non_empty_smiles:
            all_drug_ids.add(item[0])
            all_drug_ids.add(item[1])
        else:
            # check on valid smiles
            smile_idx1 = 2
            smile_idx2 = 3
            if len(item) > 12000:
                smile_idx1 = -4
                smile_idx2 = -3

            if item[smile_idx1] and item[smile_idx1] != "" and isinstance(item[smile_idx1], str) and isValidSmiles(item[smile_idx1]):
                all_drug_ids.add(item[0])
            if item[smile_idx2] and item[smile_idx2] != "" and isinstance(item[smile_idx2], str) and isValidSmiles(item[smile_idx2]):
                all_drug_ids.add(item[1])                
    
    # Create the mapping dictionary
    drug_to_id = {drug_id: idx for idx, drug_id in enumerate(sorted(all_drug_ids))}
    
    # Save the mapping 
    with open(output_json, "w") as f:
        json.dump(drug_to_id, f, indent=4)
    
    return drug_to_id

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create a dataset containing drugs mapping as requested by MSDAFL.")
    parser.add_argument('ext_dataset_path', type=str, help='Path of external datasets.')
    parser.add_argument('output_json', type=str, help='Pickle path to save the mapping dataset.')
    parser.add_argument("--non_empty_smiles", action="store_true", default=False)

    args = parser.parse_args()

    create_drug_mapping(args.ext_dataset_path, args.output_json, args.non_empty_smiles)
