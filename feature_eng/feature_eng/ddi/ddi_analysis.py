#
# Analyze the DDI in the specific drug dataset
#
import pickle
import argparse
from collections import defaultdict

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

def main(pickle_file):
    # Load the pickle file
    with open(pickle_file, 'rb') as f:
        drugs = pickle.load(f)

    # Dictionary to track interactions 
    interactions = defaultdict(set)

    # Count the number of drugs with interactions and the total number of interactions
    total_drugs_with_interactions = 0
    total_interactions = 0

    # Analyze the DDI 
    for drug in drugs:
        drug_id = drug.get('drugbank_id')
        if 'drug_interactions' in drug and drug['drug_interactions']:
            total_drugs_with_interactions += 1
            for interaction in drug['drug_interactions']:
                inter_drug_id = interaction['drugbank_id']
                interactions[drug_id].add(inter_drug_id)
                total_interactions += 1

    # Count reciprocal interactions 
    reciprocal_interactions = set()
    for drug_id in list(interactions.keys()):  # Use a list of keys to avoid updates during iterations
        interacting_drugs = interactions[drug_id]
        for inter_drug_id in interacting_drugs:
            if drug_id in interactions[inter_drug_id]:
                s = ""
                if drug_id < inter_drug_id:
                    s = drug_id
                else:
                    s = inter_drug_id 
                reciprocal_interactions.add(s)


    # Count the average number of iteraction per drug 
    average_interactions_per_drug = total_interactions / len(drugs) if drugs else 0

    # Print the results 
    print(f"Number of drugs with reciprocal interactions: {len(reciprocal_interactions)}")
    print(f"Number of drugs with at leat an interaction: {total_drugs_with_interactions}")
    print(f"Average number of interaction per drug: {average_interactions_per_drug:.2f}")
    print(f"Original Total interactions: {total_interactions}")


    print("Total number of drugs:", len(drugs))

    for drug in drugs:
        smile = drug.get('calc_prop_smiles','')
        if isinstance(smile, float):
            smile = ''
        drug['calc_prop_smiles'] = smile

    # First, we filter data that belongs to the right groups
    drugs = [
        {key: drug[key] for key in ['drugbank_id', 'targets', 'calc_prop_smiles', 'drug_interactions'] if key in drug}
        for drug in drugs if has_approved_group(drug)
    ]

    print("Total number of drugs after filtering groups:", len(drugs), len(drugs))

    # Calculate the drugs without smiles
    total_no_smiles = sum(1 for drug in drugs if not drug.get('calc_prop_smiles'))
    print("Total number of drugs without SMILES:", total_no_smiles)



    known_interactions = set()
    drug_dict = {drug['drugbank_id']: drug for drug in drugs}
    # Populate the known interactions 
    for drug in drugs:
        if 'drug_interactions' in drug:
            for interaction in drug['drug_interactions']:
                if drug['drugbank_id'] in drug_dict and interaction['drugbank_id'] in drug_dict:
                    known_interactions.add((drug['drugbank_id'], interaction['drugbank_id']))

    print(f"Total interactions after filtering groups: {len(known_interactions)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Analyze DDI.')
    parser.add_argument('pickle_file', type=str, help='Drugbank dataset.')
    args = parser.parse_args()
    main(args.pickle_file)