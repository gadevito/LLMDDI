#
# The script trains, validates, and tests MSDAFL on the datasets used in our study.
#
import sys
import os
import torch
import json
import torch.nn as nn
import datetime
import random
import numpy as np
import argparse
from sklearn import metrics
from ddi_dataset import DDIDataset, BatchLoader
from ddi_dataset_wo_type import DDIDataset as DDIDataset_WT
from ddi_dataset_wo_type import BatchLoader_EXT
from torch.utils.data import DataLoader


print_all = False
# Calculate metrics for the prediction
def calc_metrics(y_pred, y_true):
    y_pred = y_pred.numpy()
    y_true = y_true.numpy()

    y_pred_label = (y_pred >= 0.5).astype(np.int32)

    acc = metrics.accuracy_score(y_true, y_pred_label)
    auc = metrics.roc_auc_score(y_true, y_pred)
    f1 = metrics.f1_score(y_true, y_pred_label, zero_division=0)

    p = metrics.precision_score(y_true, y_pred_label, zero_division=0)
    r = metrics.recall_score(y_true, y_pred_label, zero_division=0)
    ap = metrics.average_precision_score(y_true, y_pred)

    return acc, auc, f1, p, r, ap

def loadJsonFile(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
    return data

# Get the drop rate statistics
def get_drop_rate_stats(drop_rate_list):
    drop_rate_stats = {
        "max": 0.0,
        "min": 0.0,
        "mean": 0.0
    }

    if len(drop_rate_list) == 0:
        return drop_rate_stats

    drop_rate_stats["max"] = max(drop_rate_list)
    drop_rate_stats["min"] = min(drop_rate_list)
    drop_rate_stats["mean"] = sum(drop_rate_list) / len(drop_rate_list)

    return drop_rate_stats

# CustomDataset to handle better data_paths
class CustomDataset(DDIDataset_WT):
    """Dataset personalizzato che accetta un percorso di dati specifico"""
    def __init__(self, data_path, mode="train", non_empty_smiles=False):
        self.data_path = data_path
        self.mode = mode
        
        features_file = "node_features.npy"
        interaction_file = f"{mode}.json"
        graph_file = "graphs.pt"
        smiles = "ext_drug_smiles.json"
        if os.path.isdir(data_path):
            features_file = os.path.join(data_path, features_file)
            graph_file = os.path.join(data_path, graph_file)
            interaction_file = os.path.join(data_path, interaction_file)
            smiles = os.path.join(data_path, smiles)
        else:
            directory = os.path.dirname(data_path)
            features_file = os.path.join(directory, features_file)
            graph_file = os.path.join(directory, graph_file)
            smiles = os.path.join(directory, smiles)
            interaction_file = os.path.basename(data_path)
            interaction_file = os.path.join(directory, interaction_file)

        print(interaction_file)    




        # Load the node features 
        self.node_features = np.load(features_file)
        
        # Load drug interactions 
        self.interaction_data = []
        
        if os.path.exists(interaction_file):
            import json
            with open(interaction_file, 'r') as f:
                self.interaction_data = json.load(f)

        if non_empty_smiles:
            smiles_data = loadJsonFile(smiles)
            valid_drug_indices = set()
            for i, smiles in enumerate(smiles_data):
                if smiles and smiles != "":  # Controlla che non sia None, vuoto o stringa vuota
                    valid_drug_indices.add(i)

            # Filter interaction_data maintaining only interaction where drugs have smiles
            filtered_interaction_data = []
            for interaction in self.interaction_data:
                drug1_id, drug2_id = interaction[0], interaction[1]
                if drug1_id in valid_drug_indices and drug2_id in valid_drug_indices:
                    filtered_interaction_data.append(interaction)

            self.interaction_data = filtered_interaction_data

        # Load Graphs
        self.graphs = torch.load(graph_file)
        
        self.shuffle_idx = list(range(len(self.interaction_data)))
    
    def do_shuffle(self):
        """Shuffle data"""
        random.shuffle(self.shuffle_idx)
    
    def __len__(self):
        return len(self.interaction_data)
    
    def __getitem__(self, idx):
        idx = self.shuffle_idx[idx]
        data = self.interaction_data[idx]
        
        return data


@torch.no_grad()
def evaluate(model, loader, set_len, device):
    model.eval()
    cur_num = 0
    y_pred_all, y_true_all = [], []

    for batch in loader:
        graph_batch_1, graph_batch_2, ddi_type, y_true = batch

        y_pred = model.forward_func(graph_batch_1, graph_batch_2, ddi_type)

        # Calculate the probability sigmoid for the binary classification 
        sigmoid_pred = y_pred.sigmoid()
        y_pred_all.append(sigmoid_pred.cpu())
        y_true_all.append(torch.LongTensor(y_true))

        # Convert prediction to 0 and 1 
        binary_pred = (sigmoid_pred > 0.5).long()
        cur_num += len(y_true)
        sys.stdout.write(f"\r{cur_num} / {set_len}")
        sys.stdout.flush()

    y_pred = torch.cat(y_pred_all)
    y_true = torch.cat(y_true_all)
    return calc_metrics(y_pred, y_true)


def train_custom(model, args, train_path, valid_path, test_paths):
    """
    Train, validate and test the model 
    
    Args:
        model: Model to train
        args: Command line arguments
        train_path: Path of the training set 
        valid_path: Path of the validation set 
        test_paths: Path list of the test sets 
    """
    # Load Training and Validation sets
    train_set = CustomDataset(train_path, "train", args.non_empty_smiles)
    valid_set = CustomDataset(valid_path, "valid", args.non_empty_smiles)
    
    # Load Test sets
    test_sets = [CustomDataset(test_path, test_path, args.non_empty_smiles) for test_path in test_paths]
    test_set_names = [os.path.basename(path) for path in test_paths]
    
    # Load the batch loader 
    if args.has_ddi_types:
        batch_loader = BatchLoader(args)
        model.forward_func = model.forward_transductive
    else:
        batch_loader = BatchLoader_EXT(args)
        model.forward_func = model.forward_wo_type
    
    # Dimensional information
    train_set_len = len(train_set)
    valid_set_len = len(valid_set)
    test_set_lens = [len(test_set) for test_set in test_sets]
    
    print(f"train_set_len {train_set_len} valid_set_len {valid_set_len} test_set_lens {test_set_lens}")

    # Create data loaders
    train_loader = DataLoader(
        train_set, args.batch_size, True,
        collate_fn=batch_loader.collate_fn
    )
    valid_loader = DataLoader(
        valid_set, args.batch_size, False,
        collate_fn=batch_loader.collate_fn
    )
    test_loaders = [
        DataLoader(test_set, args.batch_size, False, collate_fn=batch_loader.collate_fn)
        for test_set in test_sets
    ]
    
    # Define criterion, optimizer and scheduler
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda epoch: (1.0 if epoch < 200 else 0.1),
        last_epoch=args.start_epoch - 1
    )
    
    # Trace performance
    max_valid_acc = 0.0
    max_test_accs = [0.0] * len(test_sets)
    
    # Crea log file
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    random_number = random.randint(1000, 9999)
    os.makedirs("results", exist_ok=True)
    filename = f'test_results_{timestamp}_{random_number}.txt'
    
    # Directory for models
    os.makedirs("model", exist_ok=True)
    
    # Training loop
    with open(filename, 'a') as file:
        # Write initial log information 
        file.write(f"Training con:\n")
        file.write(f"- Train: {train_path}\n")
        file.write(f"- Valid: {valid_path}\n")
        for i, test_path in enumerate(test_paths):
            file.write(f"- Test {i+1}: {test_path}\n")
        file.write("\n")
        
        for epoch in range(args.num_epoch):
            print(f"Epoch: {args.start_epoch + epoch}")
            
            # --- TRAINING ---
            model.train()
            train_loss = 0.0
            cur_num = 0
            y_pred_all, y_true_all = [], []
            train_set.do_shuffle()
            model.drop_rate_list.clear()
            
            for i, batch in enumerate(train_loader):
                graph_batch_1, graph_batch_2, ddi_type, y_true = batch
                y_true = torch.Tensor(y_true).to(args.device)
                
                y_pred = model.forward_func(graph_batch_1, graph_batch_2, ddi_type)
                loss = criterion(y_pred, y_true)
                train_loss += loss.item()
                
                y_pred_all.append(y_pred.detach().sigmoid().cpu())
                y_true_all.append(y_true.detach().long().cpu())
                
                # Dropout statistics 
                dr_stats = get_drop_rate_stats(model.drop_rate_list)
                dr_stats_print = [f"{val:.4f}" for val in dr_stats.values()]
                dr_stats_print = ", ".join(dr_stats_print)
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                cur_num += len(y_true)
                
                sys.stdout.write(
                    f"\r{cur_num} / {train_set_len}, "
                    f"{(train_loss / (i + 1)):.6f}, "
                    f"{dr_stats_print}"
                    "          "
                )
                sys.stdout.flush()
            
            # Calculate training metrics
            y_pred = torch.cat(y_pred_all)
            y_true = torch.cat(y_true_all)
            train_acc, train_auc, train_f1, train_p, train_r, train_ap = \
                calc_metrics(y_pred, y_true)
            print()
            print(
                f"Train ACC: {train_acc:.4f}, Train AUC: {train_auc:.4f}, Train F1: {train_f1:.4f}\n"
                f"Train P:   {train_p:.4f}, Train R:   {train_r:.4f}, Train AP: {train_ap:.4f}"
            )
            
            # --- VALIDATION ---
            try:
                valid_acc, valid_auc, valid_f1, valid_p, valid_r, valid_ap = \
                    evaluate(model, valid_loader, valid_set_len, args.device)
                print()
                print(
                    f"Valid ACC: {valid_acc:.4f}, Valid AUC: {valid_auc:.4f}, Valid F1: {valid_f1:.4f}\n"
                    f"Valid P:   {valid_p:.4f}, Valid R:   {valid_r:.4f}, Valid AP: {valid_ap:.4f}"
                )
            except Exception as ee:
                continue

            # --- TEST on all TEST SETS ---
            test_metrics = []
            if print_all or valid_acc >= max_valid_acc:
                for test_idx, (test_loader, test_len, test_name) in enumerate(zip(test_loaders, test_set_lens, test_set_names)):
                    try:
                        test_acc, test_auc, test_f1, test_p, test_r, test_ap = \
                            evaluate(model, test_loader, test_len, args.device)
                    except Exception as ee:
                        print("** ERROR **",ee)
                        continue
                    test_metrics.append((test_acc, test_auc, test_f1, test_p, test_r, test_ap))
                    
                    print()
                    print(
                        f"Test {test_name} ACC:  {test_acc:.4f}, AUC:  {test_auc:.4f}, F1:  {test_f1:.4f}\n"
                        f"Test {test_name} P:    {test_p:.4f}, R:    {test_r:.4f}, AP:  {test_ap:.4f}"
                    )
                    
                    # Log the results 
                    file.write(
                        f"Epoch {args.start_epoch + epoch} - {test_name}: "
                        f"ACC: {test_acc:.4f}, AUC: {test_auc:.4f}, F1: {test_f1:.4f}, "
                        f"P: {test_p:.4f}, R: {test_r:.4f}, AP: {test_ap:.4f}\n"
                    )
                    
                    # Update the best results t
                    if test_acc > max_test_accs[test_idx]:
                        max_test_accs[test_idx] = test_acc
                        file.write(f"* NEW RECORD for {test_name} *\n")
            
            # Save model if validation is better 
            if valid_acc >= max_valid_acc:
                max_valid_acc = valid_acc
                model_path = f"model/model_{timestamp}_{random_number}_epoch{args.start_epoch + epoch}.pt"
                torch.save(model.state_dict(), model_path)
                print(f"BEST VALID IN EPOCH {args.start_epoch + epoch} - Saved at {model_path}")
                file.write(f"BEST VALID MODEL SAVED: {model_path}\n")
            
            # Update scheduler
            scheduler.step()
            print()


def train(model, args):
    """
    Original function (maintaned for compatibility and history)
    """
    if args.dataset == "drugbank":
        train_set = DDIDataset(args.dataset, "train", args.fold)
        valid_set = DDIDataset(args.dataset, "valid", args.fold)
        test_set = DDIDataset(args.dataset, "test", args.fold)
        batch_loader = BatchLoader(args)
        forward_func = model.forward_transductive
    else:
        train_set = DDIDataset_WT(args.dataset, "train")
        valid_set = DDIDataset_WT(args.dataset, "valid")
        test_set = DDIDataset_WT(args.dataset, "test")
        batch_loader = BatchLoader_EXT(args)
        forward_func = model.forward_wo_type
    
    train_set_len = len(train_set)
    valid_set_len = len(valid_set)
    test_set_len = len(test_set)
    
    train_loader = DataLoader(
        train_set, args.batch_size, True,
        collate_fn=batch_loader.collate_fn
    )
    valid_loader = DataLoader(
        valid_set, args.batch_size, False,
        collate_fn=batch_loader.collate_fn
    )
    test_loader = DataLoader(
        test_set, args.batch_size, False,
        collate_fn=batch_loader.collate_fn
    )

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda epoch: (1.0 if epoch < 200 else 0.1),
        last_epoch=args.start_epoch - 1
    )

    max_valid_acc, max_test_acc = 0.0, 0.0
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    random_number = random.randint(1000, 9999)
    filename = f'test_results_{timestamp}_{random_number}.txt'

    with open(filename, 'a') as file:
        for epoch in range(args.num_epoch):
            print(f"Epoch: {args.start_epoch + epoch}")

            train_loss = 0.0
            cur_num = 0
            y_pred_all, y_true_all = [], []
            train_set.do_shuffle()
            model.drop_rate_list.clear()

            model.train()
            for i, batch in enumerate(train_loader):
                graph_batch_1, graph_batch_2, ddi_type, y_true = batch
                y_true = torch.Tensor(y_true).to(args.device)

                y_pred = model.forward_func(graph_batch_1, graph_batch_2, ddi_type)
                loss = criterion(y_pred, y_true)
                train_loss += loss.item()

                y_pred_all.append(y_pred.detach().sigmoid().cpu())
                y_true_all.append(y_true.detach().long().cpu())

                dr_stats = get_drop_rate_stats(model.drop_rate_list)
                dr_stats_print = [f"{val:.4f}" for val in dr_stats.values()]
                dr_stats_print = ", ".join(dr_stats_print)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                if args.dataset == "drugbank":
                    cur_num += graph_batch_1.num_graphs // 2
                else:
                    cur_num += graph_batch_1.num_graphs

                sys.stdout.write(
                    f"\r{cur_num} / {train_set_len}, "
                    f"{(train_loss / (i + 1)):.6f}, "
                    f"{dr_stats_print}"
                    "          "
                )
                sys.stdout.flush()

            y_pred = torch.cat(y_pred_all)
            y_true = torch.cat(y_true_all)
            train_acc, train_auc, train_f1, train_p, train_r, train_ap = \
                calc_metrics(y_pred, y_true)
            print()
            print(
                f"Train ACC: {train_acc:.4f}, Train AUC: {train_auc:.4f}, Train F1: {train_f1:.4f}\n"
                f"Train P:   {train_p:.4f}, Train R:   {train_r:.4f}, Train AP: {train_ap:.4f}"
            )

            model.eval()

            valid_acc, valid_auc, valid_f1, valid_p, valid_r, valid_ap = \
                evaluate(model, valid_loader, valid_set_len)
            print()
            print(
                f"Valid ACC: {valid_acc:.4f}, Valid AUC: {valid_auc:.4f}, Valid F1: {valid_f1:.4f}\n"
                f"Valid P:   {valid_p:.4f}, Valid R:   {valid_r:.4f}, Valid AP: {valid_ap:.4f}"
            )

            test_acc, test_auc, test_f1, test_p, test_r, test_ap = \
                evaluate(model, test_loader, test_set_len)
            print()
            print(
                f"Test ACC:  {test_acc:.4f}, Test AUC:  {test_auc:.4f}, Test F1:  {test_f1:.4f}\n"
                f"Test P:    {test_p:.4f}, Test R:    {test_r:.4f}, Test AP:  {test_ap:.4f}"
            )

            if valid_acc > max_valid_acc:
                max_valid_acc = valid_acc
                torch.save(model.state_dict(), f"model/model_{args.start_epoch + epoch}.pt")
                print(f"BEST VALID IN EPOCH {args.start_epoch + epoch}")

            if test_acc > max_test_acc:
                max_test_acc = test_acc
                file.write("* ")  # Add asterisk
                #torch.save(model.state_dict(), f"model/model_{args.start_epoch + epoch}.pt")
                print(f"BEST TEST IN EPOCH {args.start_epoch + epoch}")
            file.write(
                f"Epoch {args.start_epoch + epoch}: "
                f"Test ACC: {test_acc:.4f}, Test AUC: {test_auc:.4f}, Test F1: {test_f1:.4f}, "
                f"Test P: {test_p:.4f}, Test R: {test_r:.4f}, Test AP: {test_ap:.4f}\n"
            )
            scheduler.step()

            print()

# Run custom training
def run_custom_training(args):
    """
    Main function to train the model on our datasets 
    """
    from ddi_predictor import InteractionPredictor
    
    # Create the model
    model = InteractionPredictor(args).to(args.device)
    
    # Training
    train_custom(
        model, 
        args, 
        train_path=args.train_path,
        valid_path=args.valid_path, 
        test_paths=args.test_paths
    )
    
    return model


if __name__ == "__main__":
    # Command Line Argument Parser
    parser = argparse.ArgumentParser()
    
    # Required args
    parser.add_argument("--train_path", type=str, required=True, 
                        help="Training set path")
    parser.add_argument("--valid_path", type=str, required=True,
                        help="Validation set path")
    parser.add_argument("--test_paths", nargs='+', required=True,
                        help="Test set list")
    
    # Args for model and training 
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--gnn_num_layers", type=int, default=3)
    parser.add_argument("--gnn_model", type=str, default="GIN", choices=["GCN", "GAT", "GIN"])
    parser.add_argument("--num_node_feats", type=int, required=True, 
                        help="Number of node features")
    parser.add_argument("--has_ddi_types", action="store_true", default=False,
                        help="Has ddi types")
    parser.add_argument("--num_ddi_types", type=int, default=0,
                        help="Number of DDI types")
    
    # GNN parameters
    parser.add_argument("--gat_num_heads", type=int, default=8)
    parser.add_argument("--gat_to_concat", action="store_true", default=False)
    parser.add_argument("--gin_nn_layers", type=int, default=5)
    
    # Other model parameters
    parser.add_argument("--num_patterns", type=int, default=60)
    parser.add_argument("--attn_out_residual", action="store_true", default=False)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--pred_mlp_layers", type=int, default=3)
    
    # Dropout parameters
    parser.add_argument("--sub_drop_freq", type=str, default="never",
                        choices=["half", "always", "never"])
    parser.add_argument("--sub_drop_mode", type=str, default="rand_per_graph",
                        choices=["rand_per_graph", "rand_per_batch", "biggest", "smallest"])
    
    # Training parameters
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="mps" if torch.backends.mps.is_available() else "cpu")
    parser.add_argument("--start_epoch", type=int, default=0)
    parser.add_argument("--num_epoch", type=int, default=300)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=0.0001)
    parser.add_argument("--fold", type=int, default=0)
    
    parser.add_argument("--non_empty_smiles", action="store_true", default=False)

    args = parser.parse_args()
    
    # Set seeds
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.deterministic = True
    
    # Training
    trained_model = run_custom_training(args)


"""
python train.py \
  --train_path /path/to/train_dataset \
  --valid_path /path/to/valid_dataset \
  --test_paths /path/to/test_dataset1 /path/to/test_dataset2 \
  --num_node_feats 86 \
  --num_epoch 100 \
  --gnn_model GIN \
  --hidden_dim 128
"""