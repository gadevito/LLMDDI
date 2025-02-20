#
# Read the segments from a folder and add them to a new unique pickle file
#
import os
import pickle
import argparse
from tqdm import tqdm  

def load_segments_from_folder(folder_path):
    segment_list = []
    
    # List all files in the provided folder 
    files = [f for f in os.listdir(folder_path) if f.endswith('.pkl') and os.path.isfile(os.path.join(folder_path, f))]
    
    for filename in tqdm(files, desc="Loading segments"):
        file_path = os.path.join(folder_path, filename)
        
        with open(file_path, 'rb') as f:
            segment = pickle.load(f)
            segment_list.extend(segment)  # Add the content
    
    return segment_list

def save_to_pickle(data, output_file):
    with open(output_file, 'wb') as f:
        pickle.dump(data, f)

def main():
    parser = argparse.ArgumentParser(description="Read segments from a folder and save them into a single pickle file.")
    parser.add_argument('input_folder', type=str, help='Path to the folder containing segment files')
    parser.add_argument('output_file', type=str, help='Path to the output pickle file')
    
    args = parser.parse_args()

    # Load segments
    combined_data = load_segments_from_folder(args.input_folder)

    # Save the combined list
    save_to_pickle(combined_data, args.output_file)

if __name__ == "__main__":
    main()
