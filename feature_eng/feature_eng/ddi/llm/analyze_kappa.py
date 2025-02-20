import pandas as pd
import numpy as np
from nltk.metrics.agreement import AnnotationTask
import argparse
import os
import pickle

def parse_arguments():
    parser = argparse.ArgumentParser(description='Calculate agreement metrics for classifications')
    parser.add_argument('--data_dir', type=str, required=True,
                       help='Directory containing iteration folders (I, II, III, IV, V)')
    return parser.parse_args()

def load_pickle_files(base_dir):
    iterations = ['I', 'II', 'III', 'IV', 'V']
    all_data = []
    
    for iter_num, iter_name in enumerate(iterations, start=1):
        iter_path = os.path.join(base_dir, f'{iter_name}')
        if not os.path.exists(iter_path):
            print(f"Warning: {iter_path} does not exist")
            continue
            
        # Find all pickles in the provided folder 
        pickle_files = [f for f in os.listdir(iter_path) if f.endswith('.pickle') or f.endswith('pkl')]
        
        for pickle_file in pickle_files:
            file_path = os.path.join(iter_path, pickle_file)
            try:
                with open(file_path, 'rb') as f:
                    data = pickle.load(f)
                    # Stampa la struttura dei dati per debug
                    print(f"Data structure for {file_path}:")
                    if isinstance(data, pd.DataFrame):
                        print("Columns:", data.columns.tolist())
                    else:
                        print("Type:", type(data))
                        if isinstance(data, list) and len(data) > 0:
                            print("First element type:", type(data[0]))
                            print("First element:", data[0])
                    
                    # Se i dati sono già un DataFrame
                    if isinstance(data, pd.DataFrame):
                        df = data
                    else:
                        # Se i dati sono una lista di tuple o altro
                        df = pd.DataFrame(data)
                    
                    # Crea ID concatenando le prime due colonne
                    df['id'] = df.iloc[:, 0].astype(str) + '_' + df.iloc[:, 1].astype(str)
                    df['iteration'] = iter_num
                    all_data.append(df)
            except Exception as e:
                print(f"Error loading {file_path}: {str(e)}")
                
    return pd.concat(all_data, ignore_index=True)

def main():
    args = parse_arguments()
    
    # Load all pickle files
    merged_df = load_pickle_files(args.data_dir)
    
    # Required columns are target and new_target
    required_columns = ['target', 'new_target']
    if not all(col in merged_df.columns for col in required_columns):
        print("Error: Required columns 'target' and 'new_target' not found in data")
        return

    # Remove rows with invalid values 
    merged_df = merged_df.dropna(subset=['target', 'new_target'])
    
    # Mappa le etichette di classificazione a valori numerici
    #label_mapping = {'0': 0, '1': 1}
    
    #merged_df['target'] = merged_df['target'].map(label_mapping)
    #merged_df['new_target'] = merged_df['new_target'].map(label_mapping)
    
    # Pivot for the DataFrame
    pivot_df = merged_df.pivot_table(
        index='id', 
        columns='iteration', 
        values='new_target', 
        aggfunc='first'
    ).reset_index()
    
    # Filter rows with no values 
    valid_rows = pivot_df.dropna().copy()
    
    # Prepare data for Fleiss' Kappa
    ratings = valid_rows.iloc[:, 1:].values
    
    # Reformat data for nltk.metrics.agreement
    data = []
    for item_idx, row in enumerate(ratings):
        for rater_idx, rating in enumerate(row):
            data.append((f'rater{rater_idx}', f'item{item_idx}', rating))
    
    # Calculate and interpetrate Fleiss' Kappa
    task = AnnotationTask(data)
    fleiss_kappa_score = task.multi_kappa()
    
    def interpret_fleiss_kappa(kappa):
        if kappa < 0:
            return "Poor agreement"
        elif kappa < 0.20:
            return "Slight agreement"
        elif kappa < 0.40:
            return "Fair agreement"
        elif kappa < 0.60:
            return "Moderate agreement"
        elif kappa < 0.80:
            return "Substantial agreement"
        elif kappa < 0.99:
            return "Almost perfect agreement"
        else:
            return "Perfect agreement"

    interpretation = interpret_fleiss_kappa(fleiss_kappa_score)
    print(f"Fleiss' Kappa: {fleiss_kappa_score}")
    print(f"Interpretation: {interpretation}")

if __name__ == "__main__":
    main()