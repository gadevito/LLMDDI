#
# Read the external datasets (the binary versions), remove drugs not in drugbank and create the full ddi dataset, comprising all the
# external datasets ddi
#
import pickle
import os
import argparse

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

datasets = [NDFRT_PDDI_FILE_INCHI_AND, KEGG_PDDI_FILE,  CREDIBLEMEDS_PDDI_FILE, DDICORPUS2011_PDDI_FILE_INCHI_AND, 
           DDICORPUS2013_PDDI_FILE_INCHI_AND, NLMCORPUS_PDDI_FILE_INCHI_AND, PKCORPUS_PDDI_FILE_INCHI_AND,
           ONCHIGHPRIORITY_PDDI_FILE, ONCNONINTERUPTIVE_PDDI_FILE, OSCAR_PDDI_FILE, HIV_FILE, HEP_FILE, FRENCH_FILE,
           WORLD_VISTA_AND]

def loadPickle(input_pickle):
    with open(input_pickle, 'rb') as f:
        drugs = pickle.load(f)
    return drugs

def writePickle(output_pickle, dataset):
    with open(output_pickle, 'wb') as f:
        pickle.dump(dataset, f)

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


def main(drug_bank_pickle, ext_dataset_path, out_file):
    drugbank = loadPickle(drug_bank_pickle)

    drugbank = [
        {key: drug[key] for key in ['drugbank_id', 'targets', 'calc_prop_smiles', 'drug_interactions'] if key in drug}
        for drug in drugbank if has_approved_group(drug)
    ]

    for drug in drugbank:
        smile = drug.get('calc_prop_smiles','')
        if isinstance(smile, float):
            smile = ''
        drug['calc_prop_smiles'] = smile

    # Extract the unique gene names 
    all_human_genes = set()
    for drug in drugbank:
        human_targets = get_human_targets(drug)
        all_human_genes.update(human_targets)

    # Sort genes for consistency
    all_human_genes = sorted(all_human_genes)

    # Remove drugs that do not target genes
    drugbank = [drug for drug in drugbank if get_human_targets(drug)]

    drug_dict = {drug['drugbank_id']: drug for drug in drugbank}

    known_interactions = set()
    for drug in drugbank:
        if 'drug_interactions' in drug:
            for interaction in drug['drug_interactions']:
                if drug['drugbank_id'] in drug_dict and interaction['drugbank_id'] in drug_dict:
                    known_interactions.add((drug['drugbank_id'], interaction['drugbank_id']))
    
    for d in datasets:
        file_name = os.path.join(ext_dataset_path, d)
        drugs = loadPickle(file_name)
        filtered_drugs = [
            drug for drug in drugs
            if drug['drug1'] in drug_dict and drug['drug2'] in drug_dict
        ]
        filtered_file_name = os.path.join(os.path.join(ext_dataset_path,'filtered'), d)
        writePickle(filtered_file_name, filtered_drugs)

        print(f"Number of interactions in {d}", len(filtered_drugs))
        for fd in filtered_drugs:
            known_interactions.add((fd['drug1'], fd['drug2']))

    print("Total number of interactions", len(known_interactions))
    # Now we can save all the interaction
    writePickle(out_file, known_interactions)
    print("Full interactions dataset saved!")

if __name__ == "__main__":        
    parser = argparse.ArgumentParser(description="Remove drugs from external datasets that are not in drugbank and create the full ddi dataset.")
    parser.add_argument('input_pickle', type=str, help='Pickle path of the datasets.')
    parser.add_argument('drug_bank_pickle', type=str, help='DrugBank pickle file.')
    parser.add_argument('output_pickle', type=str, help='Pickle file to save the full ddi dataset.')
    args = parser.parse_args()

    main(args.drug_bank_pickle, args.input_pickle, args.output_pickle)