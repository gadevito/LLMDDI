#
# Create the file containing the smiles formulas for all the mapped drugs 
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

def loadJsonFile(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
    return data

def create_smiles_file(ext_dataset_path, drug_ids_path, output_json, non_empty_smiles):
    """
    Create a JSON file containing drugs SMILES 
    """
    # Temporary dictionary 
    drug_to_smiles = {}
    dataset = [] 
    for filename in os.listdir(ext_dataset_path):
        file_path = os.path.join(ext_dataset_path, filename)
        if os.path.isfile(file_path):
            if file_path.find(".DS_Store") !=-1:
                continue
            p = loadPickle(file_path)
            dataset.extend(p)

    drug_to_id = loadJsonFile(drug_ids_path)

    # Collect SMILES for each drug 
    for item in dataset:
        smile_idx1 = 2
        smile_idx2 = 3
        if len(item) > 12000:
            smile_idx1 = -4
            smile_idx2 = -3

        if not non_empty_smiles:
            drug_to_smiles[item[0]] = item[smile_idx1]
            drug_to_smiles[item[1]] = item[smile_idx2]
        else:
            # check on valid smiles
            if item[smile_idx1] and item[smile_idx1] != "" and isinstance(item[smile_idx1], str) and isValidSmiles(item[smile_idx1]):
                drug_to_smiles[item[0]] = item[2]
            if item[smile_idx2] and item[smile_idx2] != "" and isinstance(item[smile_idx2], str) and isValidSmiles(item[smile_idx2]):
                drug_to_smiles[item[1]] = item[smile_idx2]   

    # Create an ordered list of SMILS based on the mapping 
    smiles_list = [""] * len(drug_to_id)
    for drug_id, smiles in drug_to_smiles.items():
        if drug_id in drug_to_id:
            idx = drug_to_id[drug_id]
            smiles_list[idx] = smiles
    
    print(smiles_list)
    # Save SMILES
    with open(output_json, "w") as f:
        json.dump(smiles_list, f)
    
    return smiles_list

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Create a dataset containing smiles as requested by MSDAFL.")
    parser.add_argument('ext_dataset_path', type=str, help='Pickle path of external dataset.')
    parser.add_argument('json_mapping_path', type=str, help='Pickle path of drugs id mapping.')
    parser.add_argument('output_json', type=str, help='Pickle path to save the dataset.')
    parser.add_argument("--non_empty_smiles", action="store_true", default=False)

    args = parser.parse_args()

    create_smiles_file(args.ext_dataset_path, args.json_mapping_path, args.output_json, args.non_empty_smiles)