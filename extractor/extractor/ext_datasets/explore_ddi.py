#
# Open the original external datasets and print the content
#
import pickle
import re
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

def getPDDIDict():
    return {
        "certainty":None,
        "contVal":None,
        "contraindication":None,
        "dateAnnotated":None,
        "ddiPkEffect":None,
        "ddiPkMechanism":None,
        "drug1":None,
        "drug2":None,
        "effectConcept":None,
        "evidence":None,
        "evidenceSource":None,
        "evidenceStatement":None,
        "evidenceType":None,
        "homepage":None,
        "label":None,
        "numericVal":None,
        "object":None,
        "objectUri":None,
        "pathway":None,
        "precaution":None,
        "precipitant":None,
        "precipUri":None,
        "researchStatement":None,
        "researchStatementLabel":None,
        "severity":None,
        "uri":None,
        "whoAnnotated":None,
        "source":None,
        "ddiType":None
        }
 
def loadPickle(filename):
    with open(filename, 'rb') as f:  # Apertura in modalità 'rb' per mantenere i bytes intatti
        return pickle.load(f, encoding='utf-8')  # Usa il parametro encoding
    
def _loadPickle(filename):
    with open(filename, 'r') as f:
        content = f.read()
        # Per utilizzare pickle con una stringa, dobbiamo convertirla ad encoded bytes
        byte_content = content.encode('utf-8')  # Converti la stringa in bytes
        return pickle.loads(byte_content)


            
if __name__ == "__main__":        
    parser = argparse.ArgumentParser(description="Explore a dataset containing drugs with interactions.")
    parser.add_argument('input_pickle', type=str, help='Pickle path of the datasets.')
    args = parser.parse_args()

    for d in datasets:
        file_name = os.path.join(args.input_pickle, d)

        print(f"Processing {file_name}")
        pkl = loadPickle(file_name)
        print(f"{file_name}: {len(pkl)}")

        drugs = set()
        for dr in pkl:
            drugs.add(dr['drug1'])
            drugs.add(dr['drug2'])

        print(f"\n DRUGS: {len(drugs)}")
        #writePDDIs(out_file, pkl)