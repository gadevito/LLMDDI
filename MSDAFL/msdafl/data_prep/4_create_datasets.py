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

def create_dataset_files(dataset_path, drug_to_id, output_file, non_empty_smiles):
    """
    Create train.json, valid.json, tests.json 
    """
    # Formatted data
    formatted_data = []
    
    dataset = loadPickle(dataset_path)

    for item in dataset:
        idx_label = -1
        if len(item) > 12000:
            idx_label = -5
        if not non_empty_smiles:
            drug1_idx = drug_to_id[item[0]]
            drug2_idx = drug_to_id[item[1]]
        else:
            # check on valid smiles
            smile_idx1 = 2
            smile_idx2 = 3
            if len(item) > 12000:
                smile_idx1 = -4
                smile_idx2 = -3
            if item[smile_idx1] and item[smile_idx1] != "" and item[smile_idx2] and item[smile_idx2] != "" and isinstance(item[smile_idx1], str) and isinstance(item[smile_idx2], str) and isValidSmiles(item[smile_idx1]) and isValidSmiles(item[smile_idx2]):
                drug1_idx = drug_to_id[item[0]]
                drug2_idx = drug_to_id[item[1]]    
            else:
                continue            
        label = item[idx_label]
        
        # Format: [drug1_idx, drug2_idx, label]
        formatted_data.append([drug1_idx, drug2_idx, label])
    
    # Save the file 
    with open(output_file, "w") as f:
        json.dump(formatted_data, f)

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Create a dataset formattet for MSDAFL.")
    parser.add_argument('pickle_dataset_path', type=str, help='External datasets path.')
    parser.add_argument('json_mapping', type=str, help='Json mapping.')
    parser.add_argument('output_path', type=str, help='Output json file.')
    parser.add_argument("--non_empty_smiles", action="store_true", default=False)

    args = parser.parse_args()

    drug_to_id = loadJsonFile(args.json_mapping)

    for filename in os.listdir(args.pickle_dataset_path):
        file_path = os.path.join(args.pickle_dataset_path, filename)
        if os.path.isfile(file_path):
            if file_path.find(".DS_Store") !=-1:
                continue
            print(f"Processing {file_path}")
            create_dataset_files(file_path, drug_to_id, os.path.join(args.output_path, "ds_"+filename).replace(".pickle",".json"), args.non_empty_smiles)