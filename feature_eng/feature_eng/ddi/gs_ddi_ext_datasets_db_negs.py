#
# Start from the external datasets, and create the dataset structure (compliant to our experiments) containing positives and negatives.
# The negative instances are extracted from an external pickle file, created from the drugbank dataset.
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
import itertools

NDFRT_PDDI_FILE_INCHI_AND = "ndfrt-mapped-ddis-inchi-and.pickle"
NDFRT_PDDI_FILE_INCHI_OR = "ndfrt-mapped-ddis-inchi-or.pickle"
KEGG_PDDI_FILE = "kegg-ddis.pickle"
CREDIBLEMEDS_PDDI_FILE = "crediblemeds-ddis.pickle"
DDICORPUS2011_PDDI_FILE_INCHI_AND = "ddicorpus2011-ddis-inchi-and.pickle"
DDICORPUS2011_PDDI_FILE_INCHI_OR = "ddicorpus2011-ddis-inchi-or.pickle"
DDICORPUS2013_PDDI_FILE_INCHI_AND = "ddicorpus2013-ddis-inchi-and.pickle"
DDICORPUS2013_PDDI_FILE_INCHI_OR = "ddicorpus2013-ddis-inchi-or.pickle"
NLMCORPUS_PDDI_FILE_INCHI_AND = "nlmcorpus-ddis-inchi-and.pickle"
NLMCORPUS_PDDI_FILE_INCHI_OR = "nlmcorpus-ddis-inchi-or.pickle"
PKCORPUS_PDDI_FILE_INCHI_AND = "pkcorpus-ddis-inchi-and.pickle"
PKCORPUS_PDDI_FILE_INCHI_OR= "pkcorpus-ddis-inchi-or.pickle"
ONCHIGHPRIORITY_PDDI_FILE = "onchighpriority-ddis.pickle"
ONCNONINTERUPTIVE_PDDI_FILE = "oncnoninteruptive-ddis.pickle"
OSCAR_PDDI_FILE = "oscar-ddis.pickle"
HIV_FILE="hiv-ddis.pickle"
HEP_FILE="hep-ddis.pickle"
FRENCH_FILE="frenchDB-ddis.pickle"
WORLD_VISTA_OR="worldvista-ddis-inchi-or.pickle"
WORLD_VISTA_AND="worldvista-ddis-inchi-and.pickle"

# List of external datasets
datasets = [NDFRT_PDDI_FILE_INCHI_AND, KEGG_PDDI_FILE,  CREDIBLEMEDS_PDDI_FILE, DDICORPUS2011_PDDI_FILE_INCHI_AND, 
           DDICORPUS2013_PDDI_FILE_INCHI_AND, NLMCORPUS_PDDI_FILE_INCHI_AND, PKCORPUS_PDDI_FILE_INCHI_AND,
           ONCHIGHPRIORITY_PDDI_FILE, ONCNONINTERUPTIVE_PDDI_FILE, OSCAR_PDDI_FILE, HIV_FILE, HEP_FILE, FRENCH_FILE,
           WORLD_VISTA_AND]

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
            for polypeptide in target.get('polypeptides', []):
                gene_name = polypeptide.get('gene_name')
                if gene_name:
                    human_genes.append(gene_name)
    return human_genes

# Create a boolean representation for unique genes populated with the ones targeted by the given drug
def create_target_vector(drug, all_genes):
    targets = get_human_targets(drug)
    return [1 if (gene in targets) else 0 for gene in all_genes]

# Check if at least a gene is targeted
def sanity_check(target_genes):
    found = False
    for t in target_genes:
        if t == 1:
            found = True
            break
    return found

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

def main(input_pickle, drugbank_pickle, output_pickle, known_ddi, neg_samples_pickle, start_index):
    # Load the DrugBank data from the pickle file
    drugs = loadPickle(drugbank_pickle)

    # Adjust smiles just for consistency
    for drug in drugs:
        smile = drug.get('calc_prop_smiles','')
        drug['calc_prop_smiles'] = smile

    # First, we filter drugs that belong to the right groups
    drugs = [
        {key: drug[key] for key in ['drugbank_id', 'targets', 'calc_prop_smiles', 'drug_interactions'] if key in drug}
        for drug in drugs if has_approved_group(drug)
    ]

    # Extract the unique gene names 
    all_human_genes = set()
    for drug in drugs:
        human_targets = get_human_targets(drug)
        all_human_genes.update(human_targets)

    print("Total genes", len(all_human_genes))
    
    # Sort genes for consistency
    all_human_genes = sorted(all_human_genes)

    # get the organisms
    all_drug_org_dict =get_organisms(drugs)

    # Preserve the original drugbank dataset containing only drugs that target at least one gene
    drugbank_drugs = [drug for drug in drugs if get_human_targets(drug)]
    drugbank_known_interactions = set()

    drugbank_drug_dict = {drug['drugbank_id']: drug for drug in drugbank_drugs}

    # Populate the drugbank interaction to avoid overlap
    for drug in drugbank_drugs:
        if 'drug_interactions' in drug:
            for interaction in drug['drug_interactions']:
                if drug['drugbank_id'] in drugbank_drug_dict and interaction['drugbank_id'] in drugbank_drug_dict:
                    drugbank_known_interactions.add((drug['drugbank_id'], interaction['drugbank_id']))

    # Load the external dataset
    dataset_drugs = loadPickle(input_pickle)

    # Filter the external dataset, selecting drugs that are in the drugbank dataset and target genes
    drugs = [ (drug["drug1"], drug["drug2"]) for drug in dataset_drugs if drug["drug1"] in drugbank_drug_dict and drug["drug2"] in drugbank_drug_dict]

    # Load the full known interactions 
    all_known_interactions = loadPickle(known_ddi) 

    # Extract all ids in the external dataset to create pairs 
    all_drug_ids = set() 
    for drug in drugs:
        all_drug_ids.add(drug[0])
        all_drug_ids.add(drug[1])

    all_drug_ids = list(all_drug_ids)

    gc.collect()

    # Create the list to store the final dataset data 
    dataset = []

    # Add Positives to the dataset
    for drug1_id, drug2_id in tqdm(drugs, desc="Generating positive instances"):
        drug1 = drugbank_drug_dict[drug1_id]
        drug2 = drugbank_drug_dict[drug2_id]

        # Avoinding overlap with the drugbank interactions
        if (drug1_id,drug2_id) in drugbank_known_interactions or (drug2_id,drug1_id) in drugbank_known_interactions:
            continue

        target_vector1 = create_target_vector(drug1, all_human_genes)
        target_vector2 = create_target_vector(drug2, all_human_genes)

        drug1_organism = all_drug_org_dict[drug1_id]
        drug2_organism = all_drug_org_dict[drug2_id]
        enc_drug1_organism = encode_organism(drug1_organism)
        enc_drug2_organism = encode_organism(drug2_organism)

        if not sanity_check(target_vector1) or not sanity_check(target_vector2):
            print("Found an interaction without targeted genes")

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
            1  # positive class
        ))
    
    gc.collect()

    # Generate true negatives 
    num_positives = len(dataset)
    print(f"Total number of TRUE POSITIVE: {num_positives}" )

    neg_instances = loadPickle(neg_samples_pickle)

    end_index = start_index + num_positives

    # Possible pair must not be in the all_know_interactions (total interactions).
    possible_pairs = neg_instances[start_index:end_index]


    with tqdm(total= num_positives, desc="Generating true negatives") as pbar:
        for d in possible_pairs:
            dataset.append(d)
            pbar.update(1)

    num_negatives = len(dataset) - num_positives
    print(f"Total number of TRUE NEGATIVE: {num_negatives}")

    # Save the dataset to the provided pickle file
    with open(output_pickle, 'wb') as f:
        pickle.dump(dataset, f)

    print("Dataset successful created and saved!")

    #true_positives_empty_smiles = sum(1 for d in dataset if d[-1] == 1 and (not d[2] or not d[3]))
    #true_negatives_empty_smiles = sum(1 for d in dataset if d[-1] == 0 and (not d[2] or not d[3]))

    #print(f"Totale TP with null SMILES:{true_positives_empty_smiles}")
    #print(f"Totale TN with null SMILES:{true_negatives_empty_smiles}")
    return end_index

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create a dataset containing drugs targeting genes with interactions and drugs targeting genes without interactions.")
    parser.add_argument('drugbank_pickle', type=str, help='Pickle file containing DrugBank data.')
    parser.add_argument('ext_dataset_path', type=str, help='Pickle path of external datasets.')
    parser.add_argument('output_pickle', type=str, help='Pickle path to save the datasets.')
    parser.add_argument('known_ddi', type=str, help='Pickle file of the full known ddi.')
    parser.add_argument('neg_samples_pickle', type=str, help='Pickle file of the full negative instances to pick from.')
    args = parser.parse_args()
    start_index = 0
    for d in datasets:
        file_name = os.path.join(args.ext_dataset_path, d)
        output_pickle = os.path.join(args.output_pickle, "ds_"+d)
        print(f"Process {file_name}")
        
        start_index = main(file_name, args.drugbank_pickle, output_pickle, args.known_ddi, args.neg_samples_pickle, start_index)
