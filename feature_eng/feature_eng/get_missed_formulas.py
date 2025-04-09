# Use external databases to get missing formulas (inchi, SMILES, weight, etc.)
import pandas as pd
import pubchempy as pcp
import argparse
from tqdm import tqdm
import requests
import traceback
import time
from Bio.KEGG import REST
from rdkit import Chem
from rdkit.Chem import Descriptors
from Bio import Entrez, SeqIO
from bs4 import BeautifulSoup
from concurrent.futures import ThreadPoolExecutor, as_completed
from urllib.error import HTTPError
from urllib import parse
# Configura l'email per Entrez
Entrez.email = "your.email@example.com"  # Sostituisci con la tua email

def retry_request(func, *args, retries=3, backoff_factor=1.0, **kwargs):
    """Helper function to retry requests with exponential backoff."""
    for attempt in range(retries):
        try:
            return func(*args, **kwargs)
        except HTTPError as e:
            if e.code == 429:  # Too Many Requests
                wait = backoff_factor * (2 ** attempt)
                print(f"429 Too Many Requests. Retrying in {wait} seconds...")
                time.sleep(wait)
            else:
                raise
    raise Exception("Max retries exceeded")

def get_kegg_id(drug_name):
    try:
        result = REST.kegg_find("drug", drug_name)
        # Analizza i risultati
        drug_ids = []
        for line in result:
            if line.startswith("dr:"):
                drug_id = line.split()[0]
                drug_ids.append(drug_id)
        return drug_ids[0]
    except Exception as e:
        return None

def fetch_kegg_data(name, key=None):
    id = key if key is not None else get_kegg_id(name) 
    if id is None:
        return None, None, None
    
    # Effettua una richiesta GET alla pagina web
    response = requests.get(f"https://www.kegg.jp/entry/{id}",timeout=30)
    response.raise_for_status()  # Controlla se ci sono stati errori nella richiesta

    # Analizza il contenuto HTML con BeautifulSoup
    soup = BeautifulSoup(response.text, 'html.parser')
    formula = None
    mol_weight = None
    sequence_result = None
    # Cerca la formula
    try:
        formula_row = soup.find('span', class_='nowrap', string='Formula').find_parent('th').find_next_sibling('td')
        formula = formula_row.find('div', class_='cel').get_text(strip=True)
    except Exception as e:
        pass

    # Cerca il peso molecolare
    try:
        mol_weight_row = soup.find('span', class_='nowrap', string='Mol weight').find_parent('th').find_next_sibling('td')
        mol_weight = mol_weight_row.find('div', class_='cel').get_text(strip=True)
    except Exception as e:
        pass

    # Cerca la sequenza
    try:
        sequence_row = soup.find('span', class_='nowrap', string='Sequence').find_parent('th').find_next_sibling('td')
        sequence_content = sequence_row.find('div', class_='cel').decode_contents()
        
        # Sostituisci i tag <br> con \n e rimuovi gli spazi
        sequence_content = sequence_content.replace('<br/>', '\n').replace(' ', '')

        # Prendi la parte di sequenza fino a che non viene trovato ":"
        sequence_lines = sequence_content.splitlines()
        sequence_until_colon = []

        for line in sequence_lines:
            if ':' in line:
                break
            if line.startswith("("):
                line = ">"+line
            sequence_until_colon.append(line)

        # Unisce le righe della sequenza
        sequence_result = '\n'.join(sequence_until_colon)
    except Exception as e:
        pass
    return sequence_result, formula, mol_weight

def get_protein_sequence_from_ncbi(protein_name, max_results=2):
    #time.sleep(2)
    handle = retry_request(Entrez.esearch, db="protein", term=protein_name, retmax=max_results)
    record = Entrez.read(handle)
    handle.close()

    fasta_sequences = []
    second = False

    for protein_id in record["IdList"]:
        handle = retry_request(Entrez.efetch, db="protein", id=protein_id, rettype="fasta", retmode="text")
        seq_record = SeqIO.read(handle, "fasta")
        handle.close()

        if second:
            fasta_header = f"\n>{seq_record.id} {seq_record.description}"
        else:
            fasta_header = f">{seq_record.id} {seq_record.description}"
        fasta_sequence = str(seq_record.seq)
        fasta_full = f"{fasta_header}\n{fasta_sequence}"
        fasta_sequences.append(fasta_full)
        second = True

    return "".join(fasta_sequences)

def get_current_accessions(deprecated_accession):
    url = f"https://rest.uniprot.org/uniprotkb/search?query=accession:{deprecated_accession}&format=json"
    response = requests.get(url,timeout=30)
    if response.status_code == 200:
        data = response.json()
        if 'results' in data and len(data['results']) > 0:
            # Extract primary and secondary accessions
            primary_accession = data['results'][0]['primaryAccession']
            secondary_accessions = data['results'][0].get('secondaryAccessions', [])
            return [primary_accession] + secondary_accessions
    else:
        raise ValueError("Error in get_current_accessions")
    return []

def search_drug_kegg(drug_name):
    """
    Cerca un farmaco nel database KEGG e restituisce le informazioni pertinenti
    """
    def formatChain(line, newLine=False):
        line = line.strip()
        if line.startswith("("):
            line = ">"+line.replace("(","").replace(")","")
            if newLine:
                line = "\n"+line+"\n"
        else:
            line = line.replace(" ","")
        return line
    fasta_sequence = None
    formula = None
    weight = None
    drug_id = None
    #print(f"searching for {drug_name}")
    # Cerca il farmaco
    try:
        result = REST.kegg_find("drug", drug_name)
    except HTTPError as e:
        if e.code == 400:  # Too Many Requests
            return None, None, None, None
        
    # Analizza i risultati
    drug_ids = []
    for line in result:
        if line.startswith("dr:"):
            drug_id = line.split()[0]
            drug_ids.append(drug_id)
    
    if not drug_ids:
        print(f"No drug_ids for {drug_name}")
        return None, None, None, None
    
    drug_info = REST.kegg_get(drug_ids[0])

    start = False
    lines = []
        
    for line in drug_info:
        if line.startswith("SEQUENCE"):
            start = True
            s = line[len("SEQUENCE"):].strip()
            s = formatChain(s)+"\n"
            lines.append(s)
        elif start:
            if line.startswith("  TYPE") or line.find(":") !=-1:
                start = False
                break
            t = line.strip()
            t = formatChain(t,True)
            lines.append(t)    

    f_lines = []
    #print("\n\n")
    for line in drug_info:
        #print(line)
        if line.startswith("FORMULA"):
            s = line[len("FORMULA"):].strip()
            i =s.find(".")
            if i !=-1:
                s = s[:i]
            f_lines.append(s)

    w_lines = []
    for line in drug_info:
        if line.lower().startswith("mol weight"):
            s = line[len("mol weight"):].strip()
            w_lines.append(s)

    if len(lines)>0:
        fasta_sequence = "".join(lines)
    if len(f_lines)>0:
        formula = "".join(f_lines)
    if len(w_lines)>0:
        weight = "".join(w_lines)
    return drug_id, fasta_sequence, formula, weight


def get_molecular_data_from_mitotox(name):
    url =f"https://www.mitotox.org/api/compounds/list?name={name}"
    response = requests.get(url, headers={"accept": "application/json"}, timeout=30)
    
    if response.status_code == 200 and response.json()['results']:
        return response.json()['results'][0]['molecular_formula'], response.json()['results'][0]['molecular_weight']
    return None, None

def get_fasta_from_uniprot(uniprot_id):
    url = f"https://rest.uniprot.org/uniprotkb/{uniprot_id}.fasta"
    response = requests.get(url,timeout=30)
    if response.status_code == 200:
        return response.text
    return None

def get_chembl_id(compound_name):
    url = f"https://www.ebi.ac.uk/chembl/api/data/molecule/search?q={compound_name}"
    response = requests.get(url, headers={"accept": "application/json"}, timeout=30)
    
    if response.status_code == 200 and response.json()['page_meta']['total_count'] > 0:
        return response.json()['molecules'][0]['molecule_chembl_id']
    return None

def get_chemical_info_from_chembl(compound_name):
    chembl_id = get_chembl_id(compound_name)
    if chembl_id:
        url = f"https://www.ebi.ac.uk/chembl/api/data/molecule/{chembl_id}"
        time.sleep(1)
        response = requests.get(url, headers={"accept": "application/json"}, timeout=30)
        if response.status_code == 200:
            data = response.json()
                
            # Initialize a dictionary to store the chemical information
            chemical_info = {}

            # Extract SMILES, InChI, and molecular formula if available
            if 'molecule_structures' in data and data['molecule_structures']:
                chemical_info['smiles'] = data['molecule_structures'].get('canonical_smiles')
                chemical_info['inchi'] = data['molecule_structures'].get('standard_inchi')
                chemical_info['molecular_formula'] = data['molecule_structures'].get('molecular_formula', None)
                chemical_info['molecular_weight'] = data['molecule_structures'].get('molecular_weight', None)
            # Check if at least one field was retrieved
            if any(chemical_info.values()):
                return chemical_info
            else:
                return None
        else:
            raise ValueError("Error in get_chemical_info_from_chembl")
    return None

def calculate_properties_from_smiles(smiles):
    try:
        mol = Chem.MolFromSmiles(smiles)
        formula = Chem.rdMolDescriptors.CalcMolFormula(mol)
        weight = Descriptors.MolWt(mol)
        return formula, weight
    except Exception as e:
        print(f"Error calculating properties for SMILES {smiles}: {e}")
    return None, None

def getDataFromPubChem(id,var):
    time.sleep(1)
    compound = pcp.get_compounds(id, var)
    if compound:
        return {'smiles':compound[0].canonical_smiles, 'inchi':compound[0].inchi, 'formula': compound[0].molecular_formula, 'weight': compound[0].molecular_weight}
    else:
        return None
    
def get_chemical_data(row):
    smiles = None
    inchi = None
    molecular_formula = None
    fasta_sequence = None
    molecular_weight = None
    processed = False
    try:
        # Check if SMILES and InChI already have values
        if pd.notnull(row['calc_prop_smiles']) and pd.notnull(row['calc_prop_inchi']) and pd.notnull(row['calc_prop_molecular_formula']) and pd.notnull(row['fasta_sequence']) and pd.notnull(row['calc_prop_molecular_weight']):
            return row['calc_prop_smiles'], row['calc_prop_inchi'], row['calc_prop_molecular_formula'], row['calc_prop_molecular_weight'], row['fasta_sequence'], True
        
        cas_number = row['cas_number'] if pd.notnull(row['cas_number']) else None
        uniprot_id = row['uniprot_id'] if pd.notnull(row['uniprot_id']) else None
        unii = row['unii'] if pd.notnull(row['unii']) else None
        name = row['name'] if pd.notnull(row['name']) else None

        smiles = row['calc_prop_smiles'] if pd.notnull(row['calc_prop_smiles']) else None
        inchi = row['calc_prop_inchi'] if pd.notnull(row['calc_prop_inchi']) else None
        molecular_formula = row['calc_prop_molecular_formula'] if pd.notnull(row['calc_prop_molecular_formula']) else None
        fasta_sequence = row['fasta_sequence'] if pd.notnull(row['fasta_sequence']) else None
        molecular_weight = row['calc_prop_molecular_weight'] if pd.notnull(row['calc_prop_molecular_weight']) else None

        # Try to find the compound using CAS in PubChem
        PubChem_tried = False
        if pd.notnull(cas_number) and (not smiles or not inchi or not molecular_formula or not molecular_weight):
            ret = getDataFromPubChem(cas_number, 'name')
            if ret:
                PubChem_tried = True
                if not smiles:
                    smiles = ret['smiles']
                if not inchi:
                    inchi = ret['inchi']
                if not molecular_formula:
                    molecular_formula = ret['formula']
                if not molecular_weight:
                    molecular_weight = ret['weight']
        
        # If not found, try using UNII in PubChem
        if pd.notnull(unii) and not PubChem_tried and (not smiles or not inchi or not molecular_formula or not molecular_weight):
            ret = getDataFromPubChem(unii, 'name')
            if ret:
                PubChem_tried = True
                if not smiles:
                    smiles = ret['smiles']
                if not inchi:
                    inchi = ret['inchi']
                if not molecular_formula:
                    molecular_formula = ret['formula']
                if not molecular_weight:
                    molecular_weight = ret['weight']

        # If not found, try using the drug name in PubChem
        if pd.notnull(name) and not PubChem_tried and (not smiles or not inchi or not molecular_formula or not molecular_weight):
            ret = getDataFromPubChem(name, 'name')
            if ret:
                PubChem_tried = True
                if not smiles:
                    smiles = ret['smiles']
                if not inchi:
                    inchi = ret['inchi']
                if not molecular_formula:
                    molecular_formula = ret['formula']
                if not molecular_weight:
                    molecular_weight = ret['weight']

        # Alternate methods if PubChem fails
        if not smiles or not inchi or not molecular_formula:
            if pd.notnull(name):
                cd = get_chemical_info_from_chembl(name)
                if cd:
                    smiles = smiles or cd.get('smiles',None)
                    inchi = inchi or cd.get('inchi',None)
                    molecular_formula = molecular_formula or cd.get('molecular_formula', None)
                    molecular_weight = molecular_weight or cd.get('molecular_weight', None)

        if pd.notnull(uniprot_id) and not fasta_sequence and not smiles:
            entered = False
            current_accessions = get_current_accessions(uniprot_id)
            for accession in current_accessions:
                entered = True
                fs = get_fasta_from_uniprot(accession)
                if fs:
                    fasta_sequence = get_fasta_from_uniprot(accession)
                    break
            if not entered:
                fasta_sequence = get_fasta_from_uniprot(uniprot_id)

        k = None
        if (not fasta_sequence or len(fasta_sequence) ==0) and not smiles:
            sc = row['kegg_drug_id'] if pd.notnull(row['kegg_drug_id']) else name
            if sc:
                if pd.isnull(row['kegg_drug_id']):
                    sc = parse.quote(name) 
                k, fasta_sequence,f, w = search_drug_kegg(sc)
                if f and not molecular_formula:
                    molecular_formula = f
                if w and not molecular_weight:
                    molecular_weight = w

        if (not fasta_sequence or len(fasta_sequence) ==0) and not smiles:
            fs = get_protein_sequence_from_ncbi(name)
            if fs:
                fasta_sequence = fs

        if not molecular_formula:
            f,w = get_molecular_data_from_mitotox(name)
            if f:
                molecular_formula = f
                molecular_weight = molecular_weight or w
        
        if not molecular_formula:
            if pd.notnull(row['kegg_drug_id']):
                molecular_formula = search_drug_kegg(row['kegg_drug_id'])
            elif name:
                molecular_formula = search_drug_kegg(parse.quote(name))

        if not fasta_sequence or not molecular_formula or not molecular_weight:
            fa, f, w = fetch_kegg_data(name, k)
            if fa and not fasta_sequence:
                fasta_sequence = fa
            if f and not molecular_formula:
                molecular_formula = f
            if w and not molecular_weight:
                molecular_weight = w

        #if not inchi:
        #    try:
        #        if pd.notnull(name):
        #            inchi = get_inchi_from_cir(name)
        #    except Exception as e:
        #        print(f"Error during CIR search {name}: {e}")

        # Calculate properites
        if smiles and (not molecular_formula or not molecular_weight):
            molecular_formula, molecular_weight = calculate_properties_from_smiles(smiles)

        if not smiles and not inchi and not molecular_formula and not fasta_sequence:
            print(f"Chemical information not found for {name} (ID: {uniprot_id})")

        processed = True
        """
        row['calc_prop_smiles'] = smiles
        row['calc_prop_inchi'] = inchi
        row['calc_prop_molecular_formula'] = molecular_formula
        row['calc_prop_molecular_weight'] = molecular_weight
        row['fasta_sequence'] = fasta_sequence
        """
    except Exception as e:
        print(f"Error processing row: {row['name']}")
        print(traceback.format_exc())
    return smiles, inchi, molecular_formula, molecular_weight, fasta_sequence, processed


def process_batch_parallel(batch_df):
    results = []
    indices = []  # Lista per mantenere traccia degli indici originali
    tot = 0
    with ThreadPoolExecutor(max_workers=5) as executor:
        # Submit each row to the executor for processing
        futures = {executor.submit(get_chemical_data, row): index for index, row in batch_df.iterrows()}
        
        for future in as_completed(futures):
            index = futures[future]
            # Get the result for each future
            processed_data = future.result()
            results.append(processed_data)
            indices.append(index)  # Conserva l'indice originale
            tot += 1 if processed_data[-1] else 0
    
    print(f"Total processed rows in current batch {tot}")
    # Convert the results back into a DataFrame
    results_df = pd.DataFrame(results, columns=['calc_prop_smiles', 'calc_prop_inchi', 'calc_prop_molecular_formula', 'calc_prop_molecular_weight', 'fasta_sequence', 'processed'])

    # Assicurati di usare gli indici originali per aggiornare batch_df
    results_df.index = indices

    # Assign the results back to the batch DataFrame using .loc
    batch_df.loc[indices, ['calc_prop_smiles', 'calc_prop_inchi', 'calc_prop_molecular_formula', 'calc_prop_molecular_weight', 'fasta_sequence', 'processed']] = results_df.values

    # Assign the results back to the batch DataFrame
    #batch_df[['calc_prop_smiles', 'calc_prop_inchi', 'calc_prop_molecular_formula', 'calc_prop_molecular_weight', 'fasta_sequence', 'processed']] = results_df

    return batch_df

def process_batch(batch_df):
    results = []
    tot = 0
    indices = []  # Lista per mantenere traccia degli indici originali
    for index, row in batch_df.iterrows():
        processed_data = get_chemical_data(row)
        tot += 1 if processed_data[-1] else 0
        results.append(processed_data)
        indices.append(index)  # Conserva l'indice originale

    print(f"Total processed rows in current batch {tot}")
    # Convert the results back into a DataFrame
    results_df = pd.DataFrame(results, columns=['calc_prop_smiles', 'calc_prop_inchi', 'calc_prop_molecular_formula', 'calc_prop_molecular_weight', 'fasta_sequence','processed'])

    # Assicurati di usare gli indici originali per aggiornare batch_df
    results_df.index = indices
    
    # Assign the results back to the batch DataFrame using .loc
    batch_df.loc[indices, ['calc_prop_smiles', 'calc_prop_inchi', 'calc_prop_molecular_formula', 'calc_prop_molecular_weight', 'fasta_sequence', 'processed']] = results_df.values

    # Assign the results back to the batch DataFrame
    #batch_df[['calc_prop_smiles', 'calc_prop_inchi', 'calc_prop_molecular_formula', 'calc_prop_molecular_weight', 'fasta_sequence','processed']] = results_df

    return batch_df

def main(csv_file, chunk_size, parallel):
    # Carica il dataset
    subset_df = pd.read_csv(csv_file, dtype={
        'calc_prop_smiles': str,
        'calc_prop_inchi': str,
        'calc_prop_molecular_formula': str,
        'fasta_sequence': str,
        'processed': bool
    })

    # Aggiungi la colonna 'processed' se non esiste
    if 'processed' not in subset_df.columns:
        subset_df['processed'] = False
    
    # Filtra solo i record non ancora processati
    unprocessed_df = subset_df[~subset_df['processed']]
    chunks = [unprocessed_df.iloc[i:i + chunk_size] for i in range(0, unprocessed_df.shape[0], chunk_size)]

    results = []
    try:
        total_processed = 0
        if parallel:
            for chunk in tqdm(chunks, total=len(chunks), desc="Parallel Processing batches"):
                result = process_batch_parallel(chunk)
                results.append(result)
                total_processed += len(result)
                # Aggiorna il dataframe originale con i risultati elaborati
                subset_df.update(result)
                if total_processed % 50 == 0:
                    # Salva il dataset aggiornato
                    subset_df.to_csv(csv_file, index=False)
                    results.clear()
        else:
            for chunk in tqdm(chunks, total=len(chunks), desc="Processing batches"):
                result = process_batch(chunk)
                results.append(result)
                total_processed += len(result)
                # Aggiorna il dataframe originale con i risultati elaborati
                subset_df.update(result)
                if total_processed % 50 == 0:
                    # Salva il dataset aggiornato
                    subset_df.to_csv(csv_file, index=False)
                    results.clear()
    except Exception as e:
        print("An error occurred during processing. Saving current progress...")
        print(traceback.format_exc())
    finally:
        # Salva il dataset aggiornato alla fine del processo
        subset_df.to_csv(csv_file, index=False)

        # Stampa le statistiche
        stats = {
            "Total initial records": subset_df['drugbank_id'].notnull().sum(),
            "Total records with calc_prop_smiles not null": subset_df['calc_prop_smiles'].notnull().sum(),
            "Total records with calc_prop_inchi not null": subset_df['calc_prop_inchi'].notnull().sum(),
            "Total records with calc_prop_molecular_formula not null": subset_df['calc_prop_molecular_formula'].notnull().sum(),
            "Total records with calc_prop_molecular_weight not null": subset_df['calc_prop_molecular_weight'].notnull().sum(),
            "Total records with fasta_sequence not null": subset_df['fasta_sequence'].notnull().sum(),
            "Total records with at least one of fasta_sequence, calc_prop_inchi, or calc_prop_molecular_formula not null":
                subset_df[['calc_prop_smiles', 'fasta_sequence', 'calc_prop_inchi', 'calc_prop_molecular_formula']].notnull().any(axis=1).sum(),
            "Total records processed": subset_df['processed'].sum(),
            "Total records with at least one of calc_prop_smiles, fasta_sequence, calc_prop_inchi, or calc_prop_molecular_formula not null or not empty":
                subset_df[['calc_prop_smiles', 'fasta_sequence', 'calc_prop_inchi', 'calc_prop_molecular_formula']].apply(lambda x: any(pd.notnull(x) & (x != '')), axis=1).sum()
        }

        for key, value in stats.items():
            print(f"{key}: {value}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Recover Missing SMILES, InChI, formulas and sequences.')
    parser.add_argument('--csv_file', type=str, required=True, help='CSV file containing missing chemical data.')
    parser.add_argument('--batch_size', type=int, default=10, help='Number of batches (default: 10)')
    parser.add_argument('--parallel', action='store_true',
                        help='Enable parallelism')
    
    args = parser.parse_args()
    main(args.csv_file, args.batch_size or 10, args.parallel)

    #print(search_drug_kegg('D03455'))
    #print(search_drug_kegg('D00573', False))

    #print(get_protein_sequence_from_ncbi("Abciximab"))

    #print(search_drug_kegg("Obiltoxaximab"))

    #print(fetch_kegg_data("Obiltoxaximab"))

    #print(search_drug_kegg("Tetracosactide"))  # TODO