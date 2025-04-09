#
# Create the graph for the mapped drugs
#
import json
import argparse
import torch
import torch.nn.functional as F
import numpy as np
from rdkit import Chem

# Original code
def get_node_feat(mol, feat_dicts_raw):
    atom_idx_to_node_idx = {}
    x = [[] for i in range(8)]
    
    for i, atom in enumerate(mol.GetAtoms()):
        atom_idx_to_node_idx[atom.GetIdx()] = i
        
        x[0].append(feat_dicts_raw[0][atom.GetSymbol()])
        x[1].append(feat_dicts_raw[1][atom.GetDegree()])
        x[2].append(feat_dicts_raw[2][atom.GetImplicitValence()])
        x[3].append(feat_dicts_raw[3][atom.GetFormalCharge()])
        x[4].append(feat_dicts_raw[4][atom.GetNumRadicalElectrons()])
        x[5].append(feat_dicts_raw[5][int(atom.GetHybridization())])
        x[6].append(feat_dicts_raw[6][atom.GetTotalNumHs()])
        x[7].append(feat_dicts_raw[7][int(atom.GetIsAromatic())])
    
    feat_dim = [len(feat) for feat in feat_dicts_raw]
    
    for i in range(8):
        cur = torch.LongTensor(x[i])
        cur = F.one_hot(cur, feat_dim[i])
        x[i] = cur
    
    x = torch.cat(x, dim=-1)
    x = x.float()
    
    return x, atom_idx_to_node_idx

# Original code
def set_feat_dict(mol, feat_dicts_raw):
    for atom in mol.GetAtoms():
        feat_dicts_raw[0].add(atom.GetSymbol())
        feat_dicts_raw[1].add(atom.GetDegree())
        feat_dicts_raw[2].add(atom.GetImplicitValence())
        feat_dicts_raw[3].add(atom.GetFormalCharge())
        feat_dicts_raw[4].add(atom.GetNumRadicalElectrons())
        feat_dicts_raw[5].add(int(atom.GetHybridization()))
        feat_dicts_raw[6].add(atom.GetTotalNumHs())
        feat_dicts_raw[7].add(int(atom.GetIsAromatic()))

# Original code
def proc_feat_dicts(feat_dict_name, feat_dicts_raw):
    dict_names = ["symbol", "deg", "valence", "charge", "electron", "hybrid", "hydrogen", "aromatic"]
    feat_dicts = []
    
    output_lines = ["{\n"]
    
    for feat_dict, dict_name in zip(feat_dicts_raw, dict_names):
        feat_dict = sorted(list(feat_dict))
        feat_dict = {item : i for i, item in enumerate(feat_dict)}
        feat_dicts.append(feat_dict)
        
        output_lines.append("  \"%s\": {\n" % (dict_name))

        for key, value in feat_dict.items():
            output_lines.append(f"    \"{key}\": {value},\n")
        
        output_lines[-1] = output_lines[-1][ : -2] + "\n"
        output_lines.append("  },\n\n")

    output_lines[-1] = output_lines[-1][ : -3] + "\n"
    output_lines.append("}\n")

    with open(feat_dict_name, "w") as f:
        f.writelines(output_lines)
    
    return feat_dicts

# Original code
def get_edge_index(mol, atom_idx_to_node_idx):
    cur_edge_index = []
    
    for bond in mol.GetBonds():
        atom_1 = bond.GetBeginAtomIdx()
        atom_2 = bond.GetEndAtomIdx()
        
        node_1 = atom_idx_to_node_idx[atom_1]
        node_2 = atom_idx_to_node_idx[atom_2]

        cur_edge_index.append([node_1, node_2])
        cur_edge_index.append([node_2, node_1])
    
    if len(cur_edge_index) > 0:
        cur_edge_index = torch.LongTensor(cur_edge_index).t()
    else:
        cur_edge_index = torch.zeros((2, 0), dtype=torch.long)
    
    return cur_edge_index

# Original code
def create_empty_graph(feat_dicts):
    """
    Create an empty graph 
    """
    # Calculate the total dimension 
    total_feat_dim = sum(len(feat_dict) for feat_dict in feat_dicts)
    
    # Create a zero-tensor for empty nodes 
    x = torch.zeros((1, total_feat_dim), dtype=torch.float)
    
    # Creat an index for empty edges 
    edge_index = torch.zeros((2, 0), dtype=torch.long)
    
    return {"x": x, "edge_index": edge_index}

# Calculate the number of node features. It is needed for training
def calculate_num_node_feats(feat_dicts_raw):
    total_features = 0
    for feat_dict in feat_dicts_raw:
        total_features += len(feat_dict)
    return total_features

# Calculate the number of node features from the graph
def calculate_num_node_feats_from_graphs(graphs):
    if graphs:
        return graphs[0]["x"].shape[1]
    return 0

def get_graphs(smiles_dataset_name, feat_dict_name, graph_output, non_empty_smiles):
    with open(smiles_dataset_name, "r") as f:
        smiles_list = json.load(f)
    
    print(f"{len(smiles_list)} drugs")
    
    feat_dicts_raw = [set() for i in range(8)]
    
    # Step 1: populate the dictionary for valid SMILES
    valid_molecules_count = 0
    invalid_molecules_indices = []
    
    for idx, smiles in enumerate(smiles_list):
        if not smiles or smiles == "":
            invalid_molecules_indices.append(idx)
            print(f"WARNING: empty SMILES at {idx}")
            continue
        
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                invalid_molecules_indices.append(idx)
                print(f"WARNING: Unable to parse SMILES '{smiles}' at {idx}")
                continue
        except Exception as ee:
            invalid_molecules_indices.append(idx)
            continue

        valid_molecules_count += 1
        set_feat_dict(mol, feat_dicts_raw)
    
    print(f"Valid molecules: {valid_molecules_count}/{len(smiles_list)}")
    
    # Process features dictionary 
    feat_dicts = proc_feat_dicts(feat_dict_name, feat_dicts_raw)
    
    # Calculate the number of node features 
    num_node_feats_dict = sum(len(feat_dict) for feat_dict in feat_dicts)
    print(f"Number of features per node: {num_node_feats_dict}")
    
    # Create an empty graph
    empty_graph = create_empty_graph(feat_dicts)
    
    graphs = []
    
    print(non_empty_smiles)
    # Step 2: create graphs for all molecules 
    for idx, smiles in enumerate(smiles_list):
        if not non_empty_smiles:
            if idx in invalid_molecules_indices:
                # For invalid molecules use an empty graph
                graphs.append(empty_graph)
                continue
        try:    
            mol = Chem.MolFromSmiles(smiles)
            
            cur_x, atom_idx_to_node_idx = get_node_feat(mol, feat_dicts)
            cur_edge_index = get_edge_index(mol, atom_idx_to_node_idx)
            
            graphs.append({"x": cur_x, "edge_index": cur_edge_index})
        except Exception as ee:
            continue
    # Check num_node_feats
    for i, graph in enumerate(graphs):
        if graph["x"].shape[1] != num_node_feats_dict:
            print(f"ERROR: Graph {i} has {graph['x'].shape[1]} features instead of {num_node_feats_dict}")
    
    torch.save(graphs, graph_output)
    
    # Generate and save node_features.npy
    all_node_features = torch.cat([torch.zeros((1, num_node_feats_dict)) if len(graph["x"]) == 0 else graph["x"] for graph in graphs], dim=0)
    np.save(graph_output.replace('.pt', '_node_features.npy'), all_node_features.numpy())
    print(f"Node features saved at {graph_output.replace('.pt', '_node_features.npy')}")
    
    return num_node_feats_dict

def save_num_node_feats(output_file, num_node_feats):
    """
    Save the number of features to a txt file
    """
    with open(output_file, 'w') as f:
        f.write(f"num_node_feats = {num_node_feats}\n")
    print(f"Number of saved features saved at {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create a dataset containing smiles as requested by MSDAFL.")
    parser.add_argument('smiles_dataset', type=str, help='Json of smiles dataset.')
    parser.add_argument('features_dataset', type=str, help='Json of features dataset.')
    parser.add_argument('output_graph', type=str, help='Torch file for the graph.')
    parser.add_argument('--save_info', type=str, help='Optional: file to save the number of features')
    parser.add_argument("--non_empty_smiles", action="store_true", default=False)

    args = parser.parse_args()

    try:
        # Generate graphs and calculate num_node_feats
        num_node_feats = get_graphs(args.smiles_dataset, args.features_dataset, args.output_graph, args.non_empty_smiles)
        
        # Print the number of node features 
        print("\n" + "="*50)
        print(f"Per il training, usa il seguente parametro:")
        print(f"--num_node_feats {num_node_feats}")
        print("="*50 + "\n")
        
        # Save the number of node features 
        if args.save_info:
            save_num_node_feats(args.save_info, num_node_feats)
    except Exception as e:
        print(f"ERROR: {str(e)}")
        import traceback
        traceback.print_exc()