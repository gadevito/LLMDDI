#
# Train a L2 Model as in the following paper
# Mei, S., Zhang, K. A machine learning framework for predicting drug–drug interactions. 
# Sci Rep 11, 17619 (2021). https://doi.org/10.1038/s41598-021-97193-8
#
# The L2 model is then save to the file system.
#
import pickle
import numpy as np
from sklearn.linear_model import (LogisticRegressionCV, LogisticRegression)

from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score, matthews_corrcoef, accuracy_score
import argparse
import joblib
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split
import os
import gc

# Load the pickle file
def load_data(pickle_path):
    with open(pickle_path, 'rb') as file:
        data = pickle.load(file)
    return data

def memory_efficient_extract_features_and_target(data, start_idx, end_idx):
    """
    Extract features and target from data using minimal memory.
    Uses uint8 (8-bit unsigned integer) for binary features.
    """
    # Pre-allocate arrays with uint8 (8-bit unsigned integer)
    X = np.zeros((len(data), end_idx - start_idx), dtype=np.uint8)
    y = np.zeros(len(data), dtype=np.uint8)
    tot =len(data)
    # Fill arrays directly without intermediate lists
    for i, record in enumerate(data):
        X[i] = record[start_idx:end_idx]
        y[i] = record[-1]
        if i % 10000 == 0:
            print(f"{i} out of {tot}")
    
    return X, y

# Extract features and target 
def extract_features_and_target(data, start_idx, end_idx):
    X = np.array([record[start_idx:end_idx] for record in data])
    y = np.array([record[-1] for record in data])
    return X, y

def save_split_datasets(train_data, val_data, save_dir):
    """Save the split datasets to the specified directory"""
    os.makedirs(save_dir, exist_ok=True)
    
    train_path = os.path.join(save_dir, 'train_data.pkl')
    val_path = os.path.join(save_dir, 'val_data.pkl')
    
    with open(train_path, 'wb') as f:
        pickle.dump(train_data, f)
    with open(val_path, 'wb') as f:
        pickle.dump(val_data, f)
    
    print(f"Saved split datasets to {save_dir}")
    return train_path, val_path

def split_training_data(data, split_percentage):
    """Split the data into training and validation sets"""
    train_data, val_data = train_test_split(
        data, 
        test_size=split_percentage,
        random_state=42,
        shuffle=True,
        stratify=[record[-1] for record in data]  # Stratify based on the target variable
    )
    return train_data, val_data

def main(training_file, validation_file, model_file, split_percentage=None, save_dir=None):
    # Load training and validation sets
    training_data = load_data(training_file)

    print("Training set loaded...", len(training_data))
    # Handle data splitting if specified
    if split_percentage is not None:
        if save_dir is None:
            print("Warning: save_dir not provided. Split datasets will not be saved.")
        
        print(f"Splitting training data with {split_percentage*100}% validation split")
        training_data, validation_data = split_training_data(training_data, split_percentage)
        
        if save_dir:
            save_split_datasets(training_data, validation_data, save_dir)
    else:
        # Use provided validation file
        print("Using provided validation file")
        validation_data = load_data(validation_file)
        print("Using provided validation file", len(validation_data))

    print("Datasets loaded")
    # Compute indexes to extract genes features
    if len(training_data[0]) < 12000:
        genes_start_idx = 8
        genes_end_idx = genes_start_idx + 3921 * 2
    else:
        genes_start_idx = 4612 # This is the start genes vector from drug1
        genes_end_idx = genes_start_idx + 3921 * 2  # This is the start genes vector from drug2

    # Extract features 
    X_train, y_train = memory_efficient_extract_features_and_target(training_data, genes_start_idx, genes_end_idx)

    training_data = None
    gc.collect()

    print("Training data transformed...")
    if len(validation_data[0]) < 12000:
        genes_start_idx = 8
        genes_end_idx = genes_start_idx + 3921 * 2
    else:
        genes_start_idx = 4612 # This is the start genes vector from drug1
        genes_end_idx = genes_start_idx + 3921 * 2  # This is the start genes vector from drug2

    X_val, y_val = extract_features_and_target(validation_data, genes_start_idx, genes_end_idx)

    validation_data = None
    gc.collect()
    print("Start training...")

    # We do not need to scale data, because they are binary
    # Otherwise, we should have used the following:
    # 
    # scaler = StandardScaler()
    # X_train_scaled = scaler.fit_transform(X_train)
    # X_val_scaled = scaler.transform(X_val)


    y_train = y_train + 1  # Converte da 0/1 a 1/2
    y_val = y_val + 1      # Converte da 0/1 a 1/2

    print("Feature statistics:")
    print("Min:", X_train.min())
    print("Max:", X_train.max())
    print("Mean:", X_train.mean())
    print("Std:", X_train.std())
    print("Number of non-zero features:", np.count_nonzero(X_train))

    constant_features = []
    all_ones_features = []
    all_zeros_features = []

    # For each feature
    for i in range(X_train.shape[1]):
        feature_values = X_train[:, i]
        if np.all(feature_values == 1):  # All zeros
            constant_features.append(i)
            all_zeros_features.append(i)
        elif np.all(feature_values == 2):  # All ones
            constant_features.append(i)
            all_ones_features.append(i)

    print("Training set shape:", X_train.shape)
    print("Number of unique values in features:", np.unique(X_train))
    print("Sample of first feature vector:", X_train[0][:20])
    print("Training data statistics:")
    print("Number of positive samples:", np.sum(y_train == 2))
    print("Number of negative samples:", np.sum(y_train == 1))
    print("Total number of genes (features/2):", X_train.shape[1]//2)
    print(f"Number of constant features: {len(constant_features)}")
    print(f"Number of features with only 1: {len(all_ones_features)}")
    print(f"Number of features with only 0: {len(all_zeros_features)}")

    # Define the range for the regularizer C [from 2^-16 to 2^16]
    # as described in the paper.
    Cs = [2**i for i in range(-16, 17)]

    cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42) #5
    # Create the model with L2 regularization and cross-validation
    model = LogisticRegressionCV(
        Cs=Cs,
        cv=cv,
        penalty='l2',
        solver= 'liblinear', #saga
        scoring='accuracy',
        max_iter=10000, # was 1000
        #n_jobs=-1,
        random_state=42,
        verbose=1
    )
    # Train the model
    model.fit(X_train, y_train)


    # After training, print Coefficients
    print("Number of non-zero coefficients:", np.count_nonzero(model.coef_))
    print("Coefficient statistics:")
    print("Min coef:", model.coef_.min())
    print("Max coef:", model.coef_.max())
    print("Mean coef:", model.coef_.mean())
    print("Std coef:", model.coef_.std())

    # Predict validation data
    y_val_pred = model.predict(X_val)
    y_val_prob = model.predict_proba(X_val)[:, 1]  # Probability for the positive class

    # Convert predictions and validation labels back to 0/1
    y_val_pred = y_val_pred - 1
    y_val = y_val - 1


    # Debug: Check some predictions
    print("Some Predictions:", y_val_pred[:10])
    print("True Values:", y_val[:10])

    # Compute metrics
    auc = roc_auc_score(y_val, y_val_prob) 
    roc_auc = roc_auc_score(y_val, y_val_pred) #this is the roc-auc using the 0.50 proba threshold that the classifier has as default
    precision = precision_score(y_val, y_val_pred)
    recall = recall_score(y_val, y_val_pred)  # Sensitivity
    f1 = f1_score(y_val, y_val_pred)
    mcc = matthews_corrcoef(y_val, y_val_pred)
    accuracy = accuracy_score(y_val, y_val_pred)

    # Print metrics 
    print(f"Validation Accuracy: {accuracy}")
    print(f"ROC-AUC (val_prob): {auc}")
    print(f"ROC-AUC (val): {roc_auc}")
    print(f"Precision: {precision}")
    print(f"Sensitivity (Recall): {recall}")
    print(f"F1 Score: {f1}")
    print(f"Matthews Correlation Coefficient (MCC): {mcc}")

    # Save the model
    joblib.dump(model, model_file)

    # So we can reload it using 
    # loaded_model = joblib.load('logistic_regression_model.pkl')
    # and use it to predict
    # y_val_pred = loaded_model.predict(X_val)

    print("Trained model saved!")

    try:
        print(f"Best C: {model.C_}")
        print(f"Scores for each C: {model.scores_}")
    except Exception as e:
        pass

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and validate the L2 Model using the training and validation sets.")
    parser.add_argument('traing_pickle', type=str, help='Pickle file containing the training dataset.')
    parser.add_argument('val_pickle', type=str, help='Pickle file containing the validation dataset.')
    parser.add_argument('output_pickle', type=str, help='Pickle file to save the trained L2 model.')
    parser.add_argument('--split_percentage', type=float, help='Percentage of data to use as validation set (0.0-1.0)',
                        default=None)
    parser.add_argument('--save_dir', type=str, help='Directory to save split datasets', default=None)

    args = parser.parse_args()

    # Validate split_percentage if provided
    if args.split_percentage is not None:
        if not 0 < args.split_percentage < 1:
            raise ValueError("split_percentage must be between 0 and 1")

    main(args.traing_pickle, args.val_pickle, args.output_pickle, args.split_percentage, args.save_dir)