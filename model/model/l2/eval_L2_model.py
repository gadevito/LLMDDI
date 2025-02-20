#
# Eval the L2 Model as in the following paper
# Mei, S., Zhang, K. A machine learning framework for predicting drug–drug interactions. 
# Sci Rep 11, 17619 (2021). https://doi.org/10.1038/s41598-021-97193-8
#
import pickle
import numpy as np
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score, matthews_corrcoef, accuracy_score, confusion_matrix
import argparse
import joblib
from sklearn.model_selection import StratifiedKFold

# Load the pickle file
def load_data(pickle_path):
    with open(pickle_path, 'rb') as file:
        data = pickle.load(file)
    return data

# Extract features and target 
def extract_features_and_target(data, start_idx, end_idx):
    X = np.array([record[start_idx:end_idx] for record in data])
    y = np.array([record[-1] for record in data])
    return X, y
    
def main(validation_file, model_file, one_two_cls):
    # Load training and validation sets
    model = joblib.load(model_file)
    validation_data = load_data(validation_file)

    # Check the validation_data structure
    if len(validation_data[0]) < 12000:
        # Here we have no embeddings so the data structure should be
        # (
        #            drug1_id,
        #            drug2_id,
        #            drug1.get('calc_prop_smiles', ''),
        #            drug2.get('calc_prop_smiles', ''),
        #            drug1_organism,
        #            drug2_organism,
        #            enc_drug1_organism,
        #            enc_drug2_organism,
        #            *target_vector1,
        #            *target_vector2,
        #            0  # True negative
        #        )
        # Therefore, genes_start_idx = 8  
        # genes_end_idx = genes_start_idx + 3921 * 2  
        genes_start_idx = 8
        genes_end_idx = genes_start_idx + 3921 * 2
    else:
        # Here we have also embeddings
        # Compute indexes to extract genes features
        genes_start_idx = 4612 # This is the start genes vector from drug1
        genes_end_idx = genes_start_idx + 3921 * 2  # This is the start genes vector from drug2
    
    
    # Extract features 
    X_val, y_val = extract_features_and_target(validation_data, genes_start_idx, genes_end_idx)

    # Predict validation data
    y_val_pred = model.predict(X_val)
    y_val_prob = model.predict_proba(X_val)[:, 1]  # Probability for the positive class

    if one_two_cls:
        y_val_pred = y_val_pred - 1     


    # Compute confusion matrix
    conf_matrix = confusion_matrix(y_val, y_val_pred)
    print("Confusion Matrix:")
    print(conf_matrix)

    # Compute metrics with average='macro' to handle multiclasses
    auc = roc_auc_score(y_val, y_val_prob) 
    roc_auc = roc_auc_score(y_val, y_val_pred) #, average='macro') 
    precision = precision_score(y_val, y_val_pred) #, average='macro')
    recall = recall_score(y_val, y_val_pred) #, average='macro')
    f1 = f1_score(y_val, y_val_pred) #, average='macro')
    mcc = matthews_corrcoef(y_val, y_val_pred)
    accuracy = accuracy_score(y_val, y_val_pred)

    # Print metrics 
    print(f"Validation Accuracy: {accuracy}")
    print(f"Precision: {precision}")
    print(f"Sensitivity (Recall): {recall}")
    print(f"F1 Score: {f1}")
    print(f"ROC-AUC (val_prob): {auc}")
    print(f"ROC-AUC (val): {roc_auc}")
    print(f"Matthews Correlation Coefficient (MCC): {mcc}")


    # Print rounded metrics
    print("\n\n============\nROUNDED METRICS ===>\n")
    print(f"Validation Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Sensitivity (Recall): {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"ROC-AUC (val_prob): {auc:.4f}")
    print(f"ROC-AUC (val): {roc_auc:.4f}")
    print(f"Matthews Correlation Coefficient (MCC): {mcc:.4f}")

    # Print predictions
    #print("\nPredictions:")
    #for i, r in enumerate(validation_data):
    #    print(f"Couple: drug1:{r[0]} drug2:{r[1]} Real class:{r[-1]} "
    #          f"Pred. Class:{y_val_pred[i]} Pred prob.:{y_val_prob[i]}")

    # Optional: print class distribution
    print("\nClass distribution:")
    print("True labels:", np.bincount(y_val))
    print("Predicted labels:", np.bincount(y_val_pred))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and validate the L2 Model using the training and validation sets.")
    parser.add_argument('model_pickle', type=str, help='Pickle file containing the trained version of the model.')
    parser.add_argument('val_pickle', type=str, help='Pickle file containing the validation dataset.')
    parser.add_argument('--one_two_cls', action='store_true', help='Use 1,2 classes')
#    parser.add_argument('output_pickle', type=str, help='Pickle file to save the results.')

    args = parser.parse_args()

    main(args.val_pickle, args.model_pickle, args.one_two_cls)