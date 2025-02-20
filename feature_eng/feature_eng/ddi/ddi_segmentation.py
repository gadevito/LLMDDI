#
# Allow to extract a random validation set and to divide in segments of a fixed dimension
# a given dataset
#
import argparse
import pickle
import os
import random
from tqdm import tqdm  
from sklearn.model_selection import train_test_split

def load_dataset(pickle_file):
    with open(pickle_file, 'rb') as f:
        data = pickle.load(f)
    return data

def split_balanced(data, validation_percentage):
    # Separate data based on label
    data_0 = [item for item in data if item[-1] == 0]
    data_1 = [item for item in data if item[-1] == 1]

    # Calculate the number of samples for the validation 
    val_size_0 = int(len(data_0) * validation_percentage)
    val_size_1 = int(len(data_1) * validation_percentage)

    # Extract validation sampes in a balanced way 
    val_0 = random.sample(data_0, val_size_0)
    val_1 = random.sample(data_1, val_size_1)

    # Create the validation set
    validation_set = val_0 + val_1
    random.shuffle(validation_set)

    # Create the training set
    train_set = [item for item in data if item not in validation_set]

    return train_set, validation_set

def save_segments(train_set, segment_size, output_folder):
    os.makedirs(output_folder, exist_ok=True)
    num_segments = len(train_set) // segment_size + (1 if len(train_set) % segment_size != 0 else 0)

    for i in tqdm(range(num_segments), desc="Saving segments"):
        segment = train_set[i * segment_size: (i + 1) * segment_size]
        segment_file = os.path.join(output_folder, f'segment_{i+1}.pkl')
        with open(segment_file, 'wb') as f:
            pickle.dump(segment, f)

def main():
    parser = argparse.ArgumentParser(description="Process a dataset and split it into segments.")
    parser.add_argument('pickle_file', type=str, help='Path to the pickle file containing the dataset')
    parser.add_argument('segment_size', type=int, help='Size of each segment in the output files')
    parser.add_argument('validation_percentage', type=float, help='Percentage of the dataset to use for validation')
    parser.add_argument('output_folder', type=str, help='Folder to save the output segment files')
    
    args = parser.parse_args()

    # Load the dataset
    data = load_dataset(args.pickle_file)

    # Divide the dataset in a balanced way
    train_set, validation_set = split_balanced(data, args.validation_percentage)

    data = None

    # Save the segments
    save_segments(train_set, args.segment_size, args.output_folder)

    # Save the validation set 
    validation_file = os.path.join(args.output_folder, 'validation_set.pkl')
    with open(validation_file, 'wb') as f:
        pickle.dump(validation_set, f)

if __name__ == "__main__":
    main()
