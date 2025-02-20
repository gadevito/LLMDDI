#
# Split the provided dataset in train and validation
#
import pickle
import argparse
from sklearn.model_selection import train_test_split

def main(input_file, train_output, val_output):
    # Load the dataset to split
    with open(input_file, 'rb') as f:
        data = pickle.load(f)

    # Extract features (X) and target (y) from the dataset
    X = [t[:-1] for t in data]
    y = [t[-1] for t in data]

    # Split data in training and validation sets stratifying by target y.
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.05, random_state=42, stratify=y
    )

    # Combine data
    train_set = [x + (y,) for x, y in zip(X_train, y_train)]
    val_set = [x + (y,) for x, y in zip(X_val, y_val)]

    # Save training set
    with open(train_output, 'wb') as f:
        pickle.dump(train_set, f)

    # Save validation set
    with open(val_output, 'wb') as f:
        pickle.dump(val_set, f)

    # Check the class distribution 
    print("Class distribution in the training set:")
    print({0: y_train.count(0), 1: y_train.count(1)})

    print("Class distribution in the validation set:")
    print({0: y_val.count(0), 1: y_val.count(1)})

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Split a pickle dataset in training and validation sets.')
    parser.add_argument('input_file', type=str, help='Pickle file to split.')
    parser.add_argument('--train_output', type=str, required=True, help='Pickle file to save the training set')
    parser.add_argument('--val_output', type=str, required=True, help='Pickle file to save the validation set')
    args = parser.parse_args()
    main(args.input_file, args.train_output, args.val_output)