#
# The following script opens the the dataset, and uses gpt-4o to classify the ddi
# 

import argparse
import pickle
import os
import time
import traceback
from prompts import *
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from sklearn.model_selection import train_test_split
from openai import OpenAI
from sklearn.metrics import (confusion_matrix, classification_report, 
                             accuracy_score, roc_auc_score, f1_score,
                             precision_score, recall_score,
                             matthews_corrcoef)
import numpy as np
import pandas as pd

client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"),)

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
    
def loadPickle(input_pickle):
    with open(input_pickle, 'rb') as f:
        r = pickle.load(f)
    return r

# Get the list of genes targeted by the drug
def get_human_targets(drug):
    human_genes = []
    if 'targets' in drug:
        for target in drug['targets']:
            #if target['organism'] == 'Humans' or  target['organism'] == 'Humans and other mammals':
                for polypeptide in target.get('polypeptides', []):
                    #org = str(polypeptide['organism'])
                    #if (org in organisms) or (org.find("Human") !=-1) or all:
                    gene_name = polypeptide.get('gene_name')
                    if gene_name:
                        human_genes.append(gene_name)
    return human_genes

# Check if the drug has been approved or experimental, but not illicit or withdrawn
def has_approved_group(d):
    gr = d['groups']
    approved_or_experimental = False
    is_withdrawn_or_illicit = False
    for g in gr:
        if g in (1,2): 
            approved_or_experimental = True
        elif g in (0,3):
            is_withdrawn_or_illicit = True
            break
    return approved_or_experimental and not is_withdrawn_or_illicit

def get_organisms(drugs):
    drug_org_dict = {} 
    for drug in drugs:
        drug_id = drug['drugbank_id']
        drug_org_dict[drug_id] = ''
        if 'targets' in drug:
            for target in drug['targets']:
                drug_org_dict[drug_id] = target['organism']
            

    return drug_org_dict

def jsonl_batch(drug):

    drug1 = drug['drug_name1']
    drug2 = drug['drug_name2']
    smiles1 = drug['smiles1']
    smiles2 = drug['smiles2']
    org1 =drug['org1']
    org2 = drug['org2']
    genes1 =", ".join(drug['genes1'])
    genes2 =", ".join(drug['genes2'])
    target = "interaction" if drug['target'] == 1 else "no interaction"
    json_data = {
                    "messages": [
                        {"role": "system", "content": SYSTEM_DDI_CLASSIFICATION},
                        {"role": "user", "content":  USER_DDI_CLASSIFICATION.format(drug1=drug1, smiles1=smiles1, org1=org1, genes1=genes1, drug2=drug2, smiles2=smiles2, org2=org2, genes2=genes2)},
                        {"role": "assistant", "content": target}
                    ]
    }

    return json_data

def create_jsonl_batch(batch):
    results = []
    for d in batch:
        result = jsonl_batch(d)
        results.append(result)
    return results

def classify(drug, model="gpt-4o",backoff_factor=1.0):
    for attempt in range(5):
        try:
            cleaned_text = None
            drug1 = drug['drug_name1']
            drug2 = drug['drug_name2']
            smiles1 = drug['smiles1']
            smiles2 = drug['smiles2']
            org1 =drug['org1']
            org2 = drug['org2']
            genes1 =", ".join(drug['genes1'])
            genes2 =", ".join(drug['genes2'])
            msg = [{"role": "system", "content": SYSTEM_DDI_CLASSIFICATION}]
            msg.append({"role": "user", "content":  USER_DDI_CLASSIFICATION.format(drug1=drug1, smiles1=smiles1, org1=org1, genes1=genes1, drug2=drug2, smiles2=smiles2, org2=org2, genes2=genes2)})

            response = client.chat.completions.create(model=model,
                                            messages=msg,
                                            seed=123,
                                            max_tokens=1000,
                                            temperature = 0)
            cleaned_text = response.choices[0].message.content
            #print(USER_DDI_CLASSIFICATION.format(drug1=drug1, smiles1=smiles1, org1=org1, genes1=genes1, drug2=drug2, smiles2=smiles2, org2=org2, genes2=genes2))
            #print(cleaned_text)
            return cleaned_text.lower()
        except Exception as e:
                print("An error occurred during processing. Saving current progress...")
                print(traceback.format_exc())
                wait = backoff_factor * (2 ** attempt)
                time.sleep(wait)
    raise Exception(f"Max retries exceeded {model}")


def classify_batch(batch, model, categories):
    batch_results = []
    for row in batch:
        answer = classify(row, model)
        for category in categories:
            if category.lower() in answer.lower():
                row['label'] = category
                batch_results.append(category)
                break
        else:
            row['label'] = "none"
            batch_results.append("none")
    return batch_results

def predict(test, model, batch_size=10, max_workers=4):
    categories = ["no interaction", "interaction"]
    y_pred = [None] * len(test)  # Prealloca un array per mantenere l'ordine

    # Divide il dataset in batch
    batches = [test[i:i + batch_size] for i in range(0, len(test), batch_size)]
    
    # Utilizzando ThreadPoolExecutor per parallelizzazione dei batch
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(classify_batch, batch, model, categories): idx
            for idx, batch in enumerate(batches)
        }

        # Usare tqdm per mostrare la barra di progresso
        for future in tqdm(as_completed(futures), total=len(batches), desc="Predicting Batches"):
            idx = futures[future]
            try:
                batch_result = future.result()
                # Calcolare l'inizio del batch
                start_idx = idx * batch_size
                y_pred[start_idx:start_idx + len(batch_result)] = batch_result
            except Exception as e:
                print(f"An error occurred at batch {idx}: {e}")
                # In caso di errore, riempi il singolo batch con "none"
                start_idx = idx * batch_size
                y_pred[start_idx:start_idx + len(batch_result)] = ["none"] * len(batch_result)

    return y_pred


def evaluate(y_true, y_pred):
    labels = ["no interaction", "interaction"]
    mapping = {label: idx for idx, label in enumerate(labels)}

    def map_func(x):
        return mapping.get(
            x, -1
        )  # Map to -1 if not found, but should not occur with correct data

    y_true_mapped = np.vectorize(map_func)(y_true)
    y_pred_mapped = np.vectorize(map_func)(y_pred)

    # Calculate accuracy

    accuracy = accuracy_score(y_true=y_true_mapped, y_pred=y_pred_mapped)
    print(f"Accuracy: {accuracy:.3f}")


    # Calculate metrics
    accuracy = accuracy_score(y_true=y_true_mapped, y_pred=y_pred_mapped)
    precision = precision_score(y_true=y_true_mapped, y_pred=y_pred_mapped, average='weighted')
    sensitivity = recall_score(y_true=y_true_mapped, y_pred=y_pred_mapped, average='weighted')
    f1 = f1_score(y_true=y_true_mapped, y_pred=y_pred_mapped, average='weighted')
    mcc = matthews_corrcoef(y_true=y_true_mapped, y_pred=y_pred_mapped)
    
    # Calculate ROC AUC
    try:
        roc_auc = roc_auc_score(y_true=y_true_mapped, y_pred=y_pred_mapped)
    except:
        roc_auc = None  # In case of single class prediction

    # Print all metrics
    print(f"Accuracy: {accuracy:.3f}")
    print(f"Precision: {precision:.3f}")
    print(f"Sensitivity/Recall: {sensitivity:.3f}")
    print(f"F1 Score: {f1:.3f}")
    if roc_auc is not None:
        print(f"ROC AUC: {roc_auc:.3f}")
    print(f"Matthews Correlation Coefficient: {mcc:.3f}")


    # Generate accuracy report

    unique_labels = set(y_true_mapped)  # Get unique labels

    for label in unique_labels:
        label_indices = [
            i for i in range(len(y_true_mapped)) if y_true_mapped[i] == label
        ]
        label_y_true = [y_true_mapped[i] for i in label_indices]
        label_y_pred = [y_pred_mapped[i] for i in label_indices]
        label_accuracy = accuracy_score(label_y_true, label_y_pred)
        print(f"Accuracy for label {labels[label]}: {label_accuracy:.3f}")


    # Generate classification report
    class_report = classification_report(
        y_true=y_true_mapped,
        y_pred=y_pred_mapped,
        target_names=labels,
        labels=list(range(len(labels))),
    )
    print("\nClassification Report:")
    print(class_report)

    # Generate confusion matrix

    conf_matrix = confusion_matrix(
        y_true=y_true_mapped, y_pred=y_pred_mapped, labels=list(range(len(labels)))
    )
    print("\nConfusion Matrix:")
    print(conf_matrix)

def stratified_split(data, test_size=0.2, random_state=42):
    # Convert the dictionary list to a DataFrame
    df = pd.DataFrame(data)
    
    # Apply stratified split
    train_df, validation_df = train_test_split(
        df, 
        test_size=test_size, 
        random_state=random_state, 
        stratify=df['target']  # Stratification is based on 'target'
    )

    # Convert DataFrames to dictionary lists 
    train_data = train_df.to_dict(orient='records')
    validation_data = validation_df.to_dict(orient='records')
    
    return train_data, validation_data

def main(model, drugbank_pickle, dataset_pickle, output_pickle, num_samples, compute_perf, batch_size, num_workers):

    drugs = loadPickle(drugbank_pickle)
    ds = loadPickle(dataset_pickle)

    # First of all, we create the right data structure containing the data we need to call the LLM
    for drug in drugs:
        smile = drug.get('calc_prop_smiles','')
        if not smile or isinstance(smile, float):
            smile = ''
        drug['calc_prop_smiles'] = smile

    # First, we filter data that belongs to the right groups
    drugs = [
        {key: drug[key] for key in ['drugbank_id', 'name', 'targets', 'calc_prop_smiles', 'drug_interactions'] if key in drug}
        for drug in drugs if has_approved_group(drug)
    ]

    all_human_genes = set()
    for drug in drugs:
        human_targets = get_human_targets(drug)
        all_human_genes.update(human_targets)

    # Sort genes for consistency
    all_human_genes = sorted(all_human_genes)

    # Remove drugs that do not target genes
    drugs = [drug for drug in drugs if get_human_targets(drug)]

    drug_org_dict = get_organisms(drugs)
    # Create a dictionary to access drugs that target genes given their ids
    drug_dict = {drug['drugbank_id']: drug for drug in drugs}

    res = []
    class_distribution = {"positive":0, "negative":0}
    for i in range(len(ds)):
        t = ds[i]
        drug1 = t[0]
        drug2 = t[1]

        cdrug1 = drug_dict[drug1]
        cdrug2 = drug_dict[drug2]
        res.append({
            "drug1": drug1,
            "drug2": drug2,
            "drug_name1": cdrug1['name'],
            "drug_name2": cdrug2['name'],
            "smiles1": cdrug1['calc_prop_smiles'],
            "smiles2": cdrug2['calc_prop_smiles'],
            "genes1": get_human_targets(cdrug1),
            "genes2": get_human_targets(cdrug2),
            "org1": drug_org_dict[drug1],
            "org2": drug_org_dict[drug2],
            "target": t[-1]
        })
        if t[-1] == 1:
            class_distribution['positive'] = class_distribution['positive']+1
        else:
            class_distribution['negative'] = class_distribution['negative']+1

    

    if num_samples != -1:
        class_distribution = {"positive":0, "negative":0}
        test_size = 1-(num_samples/len(res))
        training_size = (num_samples/len(res))
        print(f"Training set reduced to {training_size:.3f} perc.: {num_samples} / ",len(res))
        train_data, validation_data = stratified_split(res, test_size=test_size, random_state=42)
        res = train_data
        class_distribution['positive'] = sum(item['target'] == 1 for item in res)
        class_distribution['negative'] = sum(item['target'] == 0 for item in res)

    print("Class Distribution", class_distribution)
    print("Total rows", len(res))

    batches = [res[i:i + batch_size] for i in range(0, len(res), batch_size)]

    final_results = []
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        future_to_batch = {executor.submit(create_jsonl_batch, batch): batch for batch in batches}

        for future in tqdm(as_completed(future_to_batch), total=len(batches), desc="Creating jsonl batches"):
            batch_results = future.result()
            final_results.extend(batch_results)

    with open(output_pickle, 'w') as f:
        for item in final_results:
            f.write(json.dumps(item) + '\n')

    print("JSONL Dataset saved!")

    if compute_perf:
        y_pred = predict(res, model, batch_size)
        y_true = [r['target'] for r in res ]
        y_true = ["no interaction" if r ==0 else 'interaction' for r in y_true ]
        evaluate(y_true, y_pred)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create the dataset for fine-tuning GPT.")
    parser.add_argument('drugbank_pickle', type=str, help='Pickle file containing the drugbank dataset.')
    parser.add_argument('dataset_training', type=str, help='Pickle file containing the training dataset.')
    parser.add_argument('dataset_val', type=str, help='Pickle file containing the validation dataset.')
    parser.add_argument('jsonl_training_out', type=str, help='JSONL file where save the training set for fine-tuning.')
    parser.add_argument('jsonl_val_out', type=str, help='JSONL file where save the validation set for fine-tuning.')

    parser.add_argument('--model', type=str, default='gpt-4o', help='Model to use for the classification.')
    parser.add_argument('--samples', type=int, default=-1, help='Number of samples to extract from the training set.')
    parser.add_argument('--compute_perf', type=str2bool, nargs='?', const=True, default=False,
                    help='Execute classification on the validation set to compute metrics (True/False)')    
    parser.add_argument('--batch_size', type=int, default=15, help='Number of cases to process per batch.')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of parallel threads to use.')

    args = parser.parse_args()

    main(None, args.drugbank_pickle, args.dataset_training, args.jsonl_training_out, args.samples, False, args.batch_size, args.num_workers)
    main(args.model, args.drugbank_pickle, args.dataset_val, args.jsonl_val_out, -1, args.compute_perf ,args.batch_size, args.num_workers)