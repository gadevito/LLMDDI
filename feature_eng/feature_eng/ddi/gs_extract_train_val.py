#
# Extract training and validation samples from a dataset
#
import pickle
import argparse
import os
import random
from collections import defaultdict, Counter

def loadPickle(input_pickle):
    with open(input_pickle, 'rb') as f:
        r = pickle.load(f)
    return r

def writePickle(output_pickle, dataset):
    with open(output_pickle, 'wb') as f:
        pickle.dump(dataset, f)

def stratified_sample(dataset, total_samples, num_samples_training, num_samples_val, is_dict=True):
    stratified = defaultdict(list)
    for item in dataset:
        if is_dict:
            key = item['target']
        else:
            key = item[-1]
        stratified[key].append(item)

    combined_sample = []
    for group in stratified.values():
        random.shuffle(group)
        k = min(len(group), total_samples // len(stratified))
        combined_sample.extend(group[:k])

    random.shuffle(combined_sample)
    training_set = combined_sample[:num_samples_training]
    validation_set = combined_sample[num_samples_training:num_samples_training + num_samples_val]

    return training_set, validation_set

def print_class_counts(dataset, is_dict=True):
    if is_dict:
        class_counts = Counter(item['target'] for item in dataset)
    else:
        class_counts = Counter(item[-1] for item in dataset)
    
    print("Class counts:", class_counts)

def main(input_pickle, output_path, num_samples_training, num_samples_val):
    ds = loadPickle(input_pickle)

    total_samples = num_samples_training + num_samples_val
    
    # Check if the dataset is a list of dictionaries
    is_dict = isinstance(ds[0], dict) and 'target' in ds[0]
    
    # Create stratified samples without overlap
    training_set, validation_set = stratified_sample(ds, total_samples, num_samples_training, num_samples_val, is_dict)
    
    # Save to pickle files
    train_output = os.path.join(output_path, f'training_set_{num_samples_training}.pkl')
    val_output = os.path.join(output_path, f'validation_set_{num_samples_val}.pkl')
    writePickle(train_output, training_set)
    writePickle(val_output, validation_set)

    print(f"Training set saved to {train_output}")
    print(f"Validation set saved to {val_output}")

    # Print class counts
    print("Training set class distribution:")
    print_class_counts(training_set, is_dict)
    print("Validation set class distribution:")
    print_class_counts(validation_set, is_dict)

if __name__ == "__main__":        
    parser = argparse.ArgumentParser(description="Create random training and validation sets for the dataset.")
    parser.add_argument('input_pickle', type=str, help='Pickle file of the dataset.')
    parser.add_argument('output_path', type=str, help='Pickle path to save the training and validation datasets.')
    parser.add_argument('--num_samples_training', type=int, default=100, help='Number of samples for training set')
    parser.add_argument('--num_samples_val', type=int, default=100, help='Number of samples for validation set')


    args = parser.parse_args()

    main(args.input_pickle, args.output_path, args.num_samples_training, args.num_samples_val)