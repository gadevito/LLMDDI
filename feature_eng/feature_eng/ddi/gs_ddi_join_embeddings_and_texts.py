#
# Starting from a dataset containing embeddings, the script 
# adds to each tuple in the dataset, textual data for smiles and organisms related to each drug pair.
# The dataset containing embeddings has the following structure:
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
# The script adds to each tuple the following textual information:
# - smiles_text1
# - smiles_text2
# - drug_organism_text1
# - drug_organism_text2
#
# using the drugbank dataset.
#
# It also writes out the ordered list of genes names.
# 
import argparse
import pickle
import gc
from tqdm import tqdm

# List of all human-related organisms
organisms = [
    "Humans", "Human Cytomegalovirus", "Human cytomegalovirus (strain Towne)",
    "Human Immunodeficiency Virus", "Human immunodeficiency virus 1", "HIV-1", "HIV-2",
    "Human herpesvirus 1", "Human herpesvirus 1 (strain KOS)", "Human adenovirus D37",
    "Human adenovirus 19", "Human papillomavirus type 11", "Human herpesvirus 5",
    "Human respiratory syncytial virus A (strain A2)",
    "Human respiratory syncytial virus A (strain RSS-2)",
    "Human respiratory syncytial virus B (strain 18537)", "Human herpesvirus 3",
    "Hepatitis C Virus", "HBV", "HBV-D", "HBV-F", "SARS-CoV", "SARS-CoV-2",
    "Middle East respiratory syndrome-related coronavirus (isolate United Kingdom/H123990006/2012)",
    "Influenza A virus (strain A/Tokyo/3/1967 H2N2)", "Influenza A virus (strain A/Aichi/2/1968 H3N2)",
    "Influenza A virus (strain A/Chile/1/1983 H1N1)", "Influenza A virus (strain A/Ann Arbor/6/1960 H2N2)",
    "Influenza A virus (strain A/Puerto Rico/8/1934 H1N1)",
    "Influenza A virus (strain A/Mallard/Astrakhan/244/1982 H14N6)",
    "Influenza A virus (strain A/Silky Chicken/Hong Kong/SF189/2001 H5N1 genotype A)",
    "Influenza A virus (strain A/Tern/Australia/G70C/1975 H11N9)",
    "Influenza A virus (strain A/Bangkok/1/1979 H3N2)", "Influenza B virus (strain B/Beijing/1/1987)",
    "Influenza B virus (strain B/Lee/1940)", "Zaire ebolavirus (strain Mayinga-76)", "Variola virus"
]

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

# Check if the drug has been approved or experimental, but not illicit or withdrawn
def has_approved_group(d):
    gr = d['groups']
    approved_or_experimental = False
    is_withdrawn_or_illicit = False
    for g in gr:
        if g in (1, 2):
            approved_or_experimental = True
        elif g in (0, 3):
            is_withdrawn_or_illicit = True
            break
    return approved_or_experimental and not is_withdrawn_or_illicit

# Load a pickle dataset
def load_pickle(file_path):
    with open(file_path, 'rb') as file:
        return pickle.load(file)

# Write a pickle dataset 
def save_pickle(data, file_path):
    with open(file_path, 'wb') as file:
        pickle.dump(data, file)

def main(dataset_with_emb, drugbank_ds, output_file, genes_outfile):
    # Load the datasets
    dataset_with_emb = load_pickle(dataset_with_emb)
    drugs = load_pickle(drugbank_ds)

    # Preprocess smiles
    for drug in drugs:
        smile = drug.get('calc_prop_smiles', '')
        if isinstance(smile, float):
            smile = ''
        drug['calc_prop_smiles'] = smile

    # Filter the drugbank dataset extracting only the needed information
    drugs = [
        {key: drug[key] for key in ['drugbank_id', 'targets', 'calc_prop_smiles', 'drug_interactions'] if key in drug}
        for drug in drugs if has_approved_group(drug)
    ]

    gc.collect()

    # Extract the unique gene names 
    all_human_genes = set()
    for drug in drugs:
        human_targets = get_human_targets(drug)
        all_human_genes.update(human_targets)

    # Sort genes for consistency
    all_human_genes = sorted(all_human_genes)

    # Preserve the original drugbank dataset containing only drugs that target at least one gene
    drugs = [drug for drug in drugs if get_human_targets(drug)]

    # This dictionary will be used as reference for all the known drugs to process.
    all_drugs = [
        {key: drug[key] for key in ['drugbank_id', 'targets', 'calc_prop_smiles'] if key in drug}
        for drug in drugs
    ]

    all_drug_org_dict = get_organisms(all_drugs)

    # Write out the gene names
    save_pickle(all_human_genes, genes_outfile)

    # Main step
    result = []
    for entry in tqdm(dataset_with_emb, desc="Adding textual information"):
        drug_id1, drug_id2 = entry[:2]
        
        drug1_info = next((drug for drug in all_drugs if drug['drugbank_id'] == drug_id1), None)
        drug2_info = next((drug for drug in all_drugs if drug['drugbank_id'] == drug_id2), None)

        if drug1_info and drug2_info:
            drug1_smiles = drug1_info.get('calc_prop_smiles', '')
            drug2_smiles = drug2_info.get('calc_prop_smiles', '')
            drug1_organism = all_drug_org_dict.get(drug_id1, '')
            drug2_organism = all_drug_org_dict.get(drug_id2, '')

            new_entry = entry + (drug1_smiles, drug2_smiles, drug1_organism, drug2_organism)
            result.append(new_entry)

    # Save the new dataset
    save_pickle(result, output_file)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process drug datasets.")
    parser.add_argument("dataset_with_emb", help="Path to the dataset with embeddings.")
    parser.add_argument("drugbank_ds", help="Path to the drugbank dataset.")
    parser.add_argument("output_file", help="Output file for the processed dataset.")
    parser.add_argument("genes_outfile", help="Output file for the genes.")
    
    args = parser.parse_args()
    main(args.dataset_with_emb, args.drugbank_ds, args.output_file, args.genes_outfile)