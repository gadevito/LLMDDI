#
# The following script opens the the dataset, and uses a GPT model to classify the ddi
# It saves the result for further analysis.
#

import argparse
import pickle
from openai import OpenAI
import os
import time
import traceback
from prompts_fs import *
from prompts_fs_hits import *
from selector import *
from tqdm import tqdm
import json 
from concurrent.futures import ThreadPoolExecutor, as_completed

client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"),)

def loadPickle(input_pickle):
    with open(input_pickle, 'rb') as f:
        r = pickle.load(f)
    return r

def writePickle(output_pickle, dataset):  
    # Save the dataset to the provided pickle file
    with open(output_pickle, 'wb') as f:
        pickle.dump(dataset, f)

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

def classify(sel, drug, model="gpt-4o", backoff_factor=1.0, random_examples=False, num_examples=10, perturbation=False):
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

            sim = sel.find_similar_entries({"smiles1": smiles1, "smiles2":smiles2, "org1":org1, "org2":org2, "genes1":drug['genes1'], "genes2":drug['genes2']}, num_examples, random_examples)

            if perturbation:
                examples = sel.format_examples_hits(sim)
                msg = [{"role": "system", "content": SYSTEM_DDI_CLASSIFICATION_RC_SEMANTIC}]
                msg.append({"role": "user", "content":  USER_DDI_CLASSIFICATION_RC_ORDER.format(examples=examples, drug1=drug1, smiles1=smiles1, org1=org1, genes1=genes1, drug2=drug2, smiles2=smiles2, org2=org2, genes2=genes2)})
            else:
                examples = sel.format_examples(sim)
                msg = [{"role": "system", "content": SYSTEM_DDI_CLASSIFICATION_RC}]
                msg.append({"role": "user", "content":  USER_DDI_CLASSIFICATION_RC.format(examples=examples, drug1=drug1, smiles1=smiles1, org1=org1, genes1=genes1, drug2=drug2, smiles2=smiles2, org2=org2, genes2=genes2)})

            response = client.chat.completions.create(model=model,
                                            messages=msg,
                                            seed=123,
                                            max_tokens=1000,
                                            temperature = 0)
            cleaned_text = response.choices[0].message.content
            i = cleaned_text.find("```json")
            if i != -1:
                j = cleaned_text.find("```", i+4)
                cleaned_text = cleaned_text[i + 7:j]
                #print("1",cleaned_text)
                res = json.loads(cleaned_text, strict=False)
                cleaned_text = res['c']
            elif cleaned_text.find("{") !=-1:
                i = cleaned_text.find("{")
                j = cleaned_text.find("}",i+1)
                cleaned_text = cleaned_text[i:j+1]
                #print("2",cleaned_text)
                res = json.loads(cleaned_text, strict=False)
                cleaned_text = res['c']

            #print(USER_DDI_CLASSIFICATION.format(drug1=drug1, smiles1=smiles1, org1=org1, genes1=genes1, drug2=drug2, smiles2=smiles2, org2=org2, genes2=genes2))
            #print(cleaned_text)
            cleaned_text = cleaned_text.replace(".","")
            return cleaned_text.lower()
        except Exception as e:
                print("An error occurred during processing. Saving current progress...")
                print(traceback.format_exc())
                wait = backoff_factor * (2 ** attempt)
                time.sleep(wait)
    raise Exception(f"Max retries exceeded {model}")

def classify_batch(sel, model,batch, random_examples, num_examples, perturbation):
    results = []
    for d in batch:
        result = classify(sel, d, model, 1.0, random_examples, num_examples, perturbation)
        d['label'] = result
        d['new_target'] = 1 if result == "interaction" else 0
        results.append(d)
    return results

def main(model, drugbank_pickle, dataset_pickle, output_pickle, batch_size, num_workers, exclude_pkl, random_examples, num_examples, perturbation):

    drugs = loadPickle(drugbank_pickle)
    ds = loadPickle(dataset_pickle)

    print(f"Opened {dataset_pickle}", len(ds))

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

    to_exclude = set()
    if exclude_pkl:
        to_exclude = loadPickle(exclude_pkl)
        to_exclude = [(t[0],t[1]) for t in to_exclude]

    res = []
    for i in tqdm(range(len(ds)), desc="Processing data for batches"):
        t = ds[i]
        if isinstance(t, dict):
            drug1 = t["drug1"]
            drug2 = t["drug2"]
            cdrug1 = drug_dict[drug1]
            cdrug2 = drug_dict[drug2]
            res.append({
                "drug1": t["drug1"],
                "drug2": t["drug2"],
                "drug_name1": cdrug1['name'],
                "drug_name2": cdrug2['name'],
                "smiles1": cdrug1['calc_prop_smiles'],
                "smiles2": cdrug2['calc_prop_smiles'],
                "genes1": get_human_targets(cdrug1),
                "genes2": get_human_targets(cdrug2),
                "org1": drug_org_dict[drug1],
                "org2": drug_org_dict[drug2],
                "target": t["target"]
            })
        else:
            
            drug1 = t[0]
            drug2 = t[1]
            if not drug_dict.get(drug1,None) or not drug_dict.get(drug2,None):
                continue

            if (drug1,drug2) in to_exclude:
                continue

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

    del drugs
    del to_exclude
    batches = [res[i:i + batch_size] for i in range(0, len(res), batch_size)]

    sel = SimilaritySearchSelector(drugbank_pickle, "../extractor/extracted_data/ddi_datasets/samples/1k/gs_training_1000.pkl", "../extractor/extracted_data/ddi_datasets/samples/1k/few_shot_training_1000.pkl")
    final_results = []
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        future_to_batch = {executor.submit(classify_batch, sel, model, batch, random_examples, num_examples, perturbation): batch for batch in batches}

        for future in tqdm(as_completed(future_to_batch), total=len(batches), desc="Classifying batches"):
            batch_results = future.result()
            final_results.extend(batch_results)

    writePickle(output_pickle, final_results)

    print("Classified Dataset saved!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Classify the interactions in the dataset.")
    parser.add_argument('drugbank_pickle', type=str, help='Pickle file containing the drugbank dataset.')
    parser.add_argument('dataset_pickle', type=str, help='Pickle file containing the validation dataset.')
    parser.add_argument('output_pickle', type=str, help='Pickle file where save the classifications.')
    parser.add_argument('--model', type=str, default='gpt-4o', help='Model to use for the classification.')
    parser.add_argument('--batch_size', type=int, default=15, help='Number of cases to process per batch.')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of parallel threads to use.')
    parser.add_argument('--exclude_pkl', type=str, help='DDI pickle file to exclude during the evaluation', default='')
    parser.add_argument("--random_examples", action="store_true", default=False,
                        help="Select random examples")
    parser.add_argument("--perturbation", action="store_true", default=False,
                        help="Perturbate prompt and examples for similarity strategy")
    parser.add_argument('--num_examples', type=int, default=10, help='Number of examples to use.')

    args = parser.parse_args()
    main(args.model, args.drugbank_pickle, args.dataset_pickle, args.output_pickle,  args.batch_size, args.num_workers, args.exclude_pkl, args.random_examples, args.num_examples, args.perturbation)