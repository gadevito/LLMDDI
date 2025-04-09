#
# Create a dataset containing only negative samples starting from the drugbank dataset. 
# The dataset structure is as follow:
# (drugbank_id1, 
#  drugbank_id2, 
#  smiles1, 
#  smiles2, 
#  drug1_organism, 
#  drug2_organism, 
#  enc_drug1_organism, 
#  enc_drug2_organism, 
#  gene_vector1, 
#  gene_vector2, 
#  target)
# 
# where:
# - drugbank_id1 is the id of the first administered drug
# - drugbank_id2 is the id of the precipitant drug
# - smiles1 is the SMILES formula for the first drug
# - smiles2 is the SMILES formula for the second drug
# - drug1_organism is the organism targeted by the first drug
# - drug2_organism is the organism targeted by the second drug
# - enc_drug1_organism is a binary (0,1) indicating if the first drug targets a human-related organism
# - enc_drug2_organism is a binary (0,1) indicating if the second drug targets a human-related organism
# - gene_vector1 and gene_vector2 are the targeted genes: 1 if the gene is targeted by the drug, 0 otherwise
# - target, 1 if drug1 and drug2 have an interaction, 0 otherwise.
#
# gene vectors are all the unique genes that are targeted. They are lexicographically sorted, so
# their positions within the tuple respect this order.
#

import pickle
import random
import argparse
from tqdm import tqdm
import os
import gc

# List of all human-related organisms
organisms =["Humans",
    "Human Cytomegalovirus",
    "Human cytomegalovirus (strain Towne)",
    "Human Immunodeficiency Virus",
    "Human immunodeficiency virus 1",
    "HIV-1",
    "HIV-2",
    "Human herpesvirus 1",
    "Human herpesvirus 1 (strain KOS)",
    "Human adenovirus D37",
    "Human adenovirus 19",
    "Human papillomavirus type 11",
    "Human herpesvirus 5",
    "Human respiratory syncytial virus A (strain A2)",
    "Human respiratory syncytial virus A (strain RSS-2)",
    "Human respiratory syncytial virus B (strain 18537)",
    "Human herpesvirus 3",
    "Hepatitis C Virus",
    "HBV",
    "HBV-D",
    "HBV-F",
    "SARS-CoV",
    "SARS-CoV-2",
    "Middle East respiratory syndrome-related coronavirus (isolate United Kingdom/H123990006/2012)",
    "Influenza A virus (strain A/Tokyo/3/1967 H2N2)",
    "Influenza A virus (strain A/Aichi/2/1968 H3N2)",
    "Influenza A virus (strain A/Chile/1/1983 H1N1)",
    "Influenza A virus (strain A/Ann Arbor/6/1960 H2N2)",
    "Influenza A virus (strain A/Puerto Rico/8/1934 H1N1)",
    "Influenza A virus (strain A/Mallard/Astrakhan/244/1982 H14N6)",
    "Influenza A virus (strain A/Silky Chicken/Hong Kong/SF189/2001 H5N1 genotype A)",
    "Influenza A virus (strain A/Tern/Australia/G70C/1975 H11N9)",
    "Influenza A virus (strain A/Bangkok/1/1979 H3N2)",
    "Influenza B virus (strain B/Beijing/1/1987)",
    "Influenza B virus (strain B/Lee/1940)",
    "Zaire ebolavirus (strain Mayinga-76)",
    "Variola virus"]

# Binary hot encoding for the given organism: human-related or not
def encode_organism(organism):
    return 1 if organism in organisms else 0

# Return a dictionary containing drugs and the related target organism 
def get_organisms(drugs):
    drug_org_dict = {} 
    for drug in drugs:
        drug_id = drug['drugbank_id']
        drug_org_dict[drug_id] = ''
        if 'targets' in drug:
            for target in drug['targets']:
                drug_org_dict[drug_id] = target['organism']
            

    return drug_org_dict

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

# Create a boolean representation for unique genes populated with the ones targeted by the given drug
def create_target_vector(drug, all_genes):
    targets = get_human_targets(drug)
    return [1 if (gene in targets) else 0 for gene in all_genes]

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

def loadPickle(input_pickle):
    with open(input_pickle, 'rb') as f:
        r = pickle.load(f)
    return r

def main(input_pickle, output_pickle, known_ddi, neg_samples):
    # Load the DrugBank data from the pickle file
    drugs = loadPickle(input_pickle)

    # Create the list to store the final dataset data 
    dataset = []
    negs_dict = set()
    try:
        dataset = loadPickle(output_pickle)
        negs_dict = {(item[0], item[1]) for item in dataset}
    except Exception as es:
        dataset = []

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

    gc.collect()

    # Extract the unique gene names 
    all_human_genes = set()
    for drug in drugs:
        human_targets = get_human_targets(drug)
        all_human_genes.update(human_targets)

    # Sort genes for consistency
    all_human_genes = sorted(all_human_genes)

    # Remove drugs that do not target genes
    drugs = [drug for drug in drugs if get_human_targets(drug)]

    print("Total drugs targeting genes", len(drugs))
    print("Total number of unique genes:", len(all_human_genes))

    # This dictionary will be used as reference for all the known drugs to process.
    all_drugs = [
        {key: drug[key] for key in ['drugbank_id', 'targets', 'calc_prop_smiles'] if key in drug}
        for drug in drugs
    ]

    # Load the full known interactions 
    #known_interactions = set()
    all_known_interactions = loadPickle(known_ddi) #set()

    # Create a dictionary to access drugs that target genes given their ids
    drug_dict = {drug['drugbank_id']: drug for drug in drugs}

    # Create a dictionary to access all known drugs given their ids
    all_drugs_dict = {drug['drugbank_id']: drug for drug in all_drugs}

    # Populate the known interactions 
    #for drug in drugs:
    #    if 'drug_interactions' in drug:
    #        for interaction in drug['drug_interactions']:
    #            if drug['drugbank_id'] in drug_dict and interaction['drugbank_id'] in drug_dict:
    #                known_interactions.add((drug['drugbank_id'], interaction['drugbank_id']))

    # Now, we do not need 'drug_interactions' anymore, so we remove it to save memory
    drugs = [
        {key: drug[key] for key in ['drugbank_id', 'targets', 'calc_prop_smiles'] if key in drug}
        for drug in drugs
    ]

    # We reconstruct the dictionary for drugs that target genes
    drug_dict = {drug['drugbank_id']: drug for drug in drugs}

    # Extract all drugbank_ids
    all_drug_ids = list(all_drugs_dict.keys())

    gc.collect()

    
    all_drug_org_dict =get_organisms(all_drugs)


    # Generate true negatives 
    print(f"Total number of SAMPLES: {neg_samples}" )
    with tqdm(total= neg_samples, desc="Generating negatives") as pbar:
        while len(dataset) < neg_samples:
            drug1_id, drug2_id = random.sample(all_drug_ids, 2)
            if (drug1_id, drug2_id) not in all_known_interactions and (drug2_id, drug1_id) not in all_known_interactions:
                if (drug1_id, drug2_id) not in negs_dict and (drug2_id, drug1_id) not in negs_dict:
                    drug1 = all_drugs_dict[drug1_id]
                    drug2 = all_drugs_dict[drug2_id]
                    target_vector1 = create_target_vector(drug1, all_human_genes)
                    target_vector2 = create_target_vector(drug2, all_human_genes)
                    drug1_organism = all_drug_org_dict[drug1_id]
                    drug2_organism = all_drug_org_dict[drug2_id]
                    enc_drug1_organism = encode_organism(drug1_organism)
                    enc_drug2_organism = encode_organism(drug2_organism)
                    dataset.append((
                        drug1_id,
                        drug2_id,
                        drug1.get('calc_prop_smiles', ''),
                        drug2.get('calc_prop_smiles', ''),
                        drug1_organism,
                        drug2_organism,
                        enc_drug1_organism,
                        enc_drug2_organism,
                        *target_vector1,
                        *target_vector2,
                        0  # True negative
                    ))

                    pbar.update(1)

    num_negatives = len(dataset)
    print(f"Total number of TRUE NEGATIVE: {num_negatives}")

    # Save the dataset to the provided pickle file
    with open(output_pickle, 'wb') as f:
        pickle.dump(dataset, f)

    print("Dataset successful created and saved!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create a dataset containing drugs targeting genes with no interactions.")
    parser.add_argument('input_pickle', type=str, help='Pickle file containing DrugBank data.')
    parser.add_argument('output_pickle', type=str, help='Pickle file to save the dataset.')
    parser.add_argument('known_ddi', type=str, help='Pickle file of the full known ddi.')
    parser.add_argument('neg_samples', type=int, default=100, help='Number of negative samples to generate (default: 100)')

    args = parser.parse_args()

    neg_samples = args.neg_samples if args.neg_samples else 100
    main(args.input_pickle, args.output_pickle, args.known_ddi, neg_samples)
