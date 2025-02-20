#
# Open the original external datasets and convert the old pickle format to the binary pickle version
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

def writePickle(output_pickle, dataset):
    with open(output_pickle, 'wb') as f:
        pickle.dump(dataset, f)

def writePDDIs(fname,PDDIs):

    res = []
    for a in PDDIs:
         
        rgx = re.compile(" ")       
        obj = a.get('object').strip()
        obj = rgx.sub("_",obj)    
                        
        pre = a.get('precipitant').strip()
        pre = rgx.sub("_", pre)
        
        s = {"drug1": a.get('drug1').replace("http://bio2rdf.org/drugbank:",""),
             "oject":obj,
             "drug2": a.get('drug2').replace("http://bio2rdf.org/drugbank:",""),
             "precipitant": pre, 
             "certainty":a.get('certainty'),
             "contraindication":a.get('contraindication'),
             "dateAnnotated": a.get('dateAnnotated'),
             "ddiPkEffect": a.get('ddiPkEffect'),
             "ddiPkMechanism": a.get('ddiPkMechanism'),
             "effectConcept": a.get('effectConcept'),
             "homepage": a.get('homepage'),
             "label": a.get('label'),
             "numericVal": a.get('numericVal'),
             "objectUri": a.get('objectUri'),           
             "pathway": a.get('pathway'),           
             "precaution": a.get('precaution'),  
             "precipUri": a.get('precipUri'),   
             "severity": a.get('severity'),  
             "uri": a.get('uri'), 
             "whoAnnotated": a.get('whoAnnotated'), 
             "source": a.get('source'), 
             "ddiType": a.get('ddiType'),
             "evidence": a.get('evidence'),
             "evidenceSource": a.get('evidenceSource'),
             "evidenceStatement":a.get('evidenceStatement'), 
             "researchStatementLabel": a.get('researchStatementLabel'),
             "researchStatement": a.get('researchStatement')}
        res.append(s)

    print("RES", len(res), res[0])
    writePickle(fname,res)
            
if __name__ == "__main__":        
    parser = argparse.ArgumentParser(description="Create a dataset containing drugs with interactions.")
    parser.add_argument('input_pickle', type=str, help='Pickle path of the datasets.')
    parser.add_argument('output_pickle', type=str, help='Pickle path to save the dataset.')
    args = parser.parse_args()

    for d in datasets:
        file_name = os.path.join(args.input_pickle, d)
        out_file = os.path.join(args.output_pickle, d)
        print(f"Processing {file_name}")
        pkl = loadPickle(file_name)
        writePDDIs(out_file, pkl)