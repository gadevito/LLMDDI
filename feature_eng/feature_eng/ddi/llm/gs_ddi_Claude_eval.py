#
# The following script opens the the dataset, and uses Anthopic to classify the ddi
# It saves the result for further analysis.
#
import argparse
import pickle
import anthropic
import os
import time
import traceback
from prompts import *
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

client = anthropic.Anthropic(
                    # defaults to os.environ.get("ANTHROPIC_API_KEY")
                    api_key=os.getenv('ANTHROPIC_API_KEY'),
                )

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

# claude-3-5-haiku-20241022
def classify(drug, model="claude-3-5-sonnet-20241022",backoff_factor=1.0):
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
            msg =[]
            msg.append({"role": "user", "content":  USER_DDI_CLASSIFICATION.format(drug1=drug1, smiles1=smiles1, org1=org1, genes1=genes1, drug2=drug2, smiles2=smiles2, org2=org2, genes2=genes2)})
            if model.find("sonnet-4") != -1 or model.find("opus-4") !=-1:
                response = client.messages.create(
                                                model=model,
                                                max_tokens=16000,
                                                #thinking={
                                                #    "type": "enabled",
                                                #    "budget_tokens": 10000
                                                #},
                                                system=SYSTEM_DDI_CLASSIFICATION,
                                                messages=msg,
                                            )
            else:
                response = client.messages.create(
                                                model=model,
                                                max_tokens=2000,
                                                system=SYSTEM_DDI_CLASSIFICATION,
                                                messages=msg,
                                                temperature = 0
                                            )
            print(response)
            if response.content and len(response.content) > 0:
                cleaned_text = response.content[0].text
                cleaned_text = cleaned_text.replace(".","")
                if cleaned_text.lower().find("no interaction") !=-1:
                    cleaned_text = "no interaction"
                else:
                    cleaned_text = "interaction"
            else:
                cleaned_text = "no interaction"
            #print(USER_DDI_CLASSIFICATION.format(drug1=drug1, smiles1=smiles1, org1=org1, genes1=genes1, drug2=drug2, smiles2=smiles2, org2=org2, genes2=genes2))
            #print(cleaned_text)
            return cleaned_text.lower()
        except Exception as e:
                print("An error occurred during processing. Saving current progress...")
                print(traceback.format_exc())
                wait = backoff_factor * (2 ** attempt)
                time.sleep(wait)
    raise Exception(f"Max retries exceeded {model}")

def classify_batch(model, batch):
    results = []
    for d in batch:
        result = classify(d, model)
        d['label'] = result
        d['new_target'] = 1 if result == "interaction" else 0
        results.append(d)
    return results

def main(model, drugbank_pickle, dataset_pickle, output_pickle, batch_size, num_workers):

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

    batches = [res[i:i + batch_size] for i in range(0, len(res), batch_size)]

    final_results = []
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        future_to_batch = {executor.submit(classify_batch, model, batch): batch for batch in batches}

        for future in tqdm(as_completed(future_to_batch), total=len(batches), desc="Classifying batches"):
            batch_results = future.result()
            final_results.extend(batch_results)

    writePickle(output_pickle, final_results)

    print("Classified Dataset saved!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Classify the interaction in the dataset.")
    parser.add_argument('drugbank_pickle', type=str, help='Pickle file containing the drugbank dataset.')
    parser.add_argument('dataset_pickle', type=str, help='Pickle file containing the validation dataset.')
    parser.add_argument('output_pickle', type=str, help='Pickle file where save the classifications.')
    parser.add_argument('--model', type=str, default='claude-3-5-sonnet-20241022', help='Model to use for the classification.')
    parser.add_argument('--batch_size', type=int, default=15, help='Number of cases to process per batch.')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of parallel threads to use.')

    args = parser.parse_args()
    main(args.model, args.drugbank_pickle, args.dataset_pickle, args.output_pickle,  args.batch_size, args.num_workers)