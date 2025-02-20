import argparse
import pickle
from tqdm import tqdm
import os

def parse_ttd_cross_file(cross_file):
    ttd_data = []
    current_id = None

    with open(cross_file, 'r', encoding='utf-8') as file:
        for line in file:
            line = line.strip()
            parts = line.split('\t')
            if len(parts) < 3:
                continue  # Skip lines that don't have enough parts
            
            if parts[1] == "TTDDRUID":
                current_id = parts[2]
                data = {'TTDDRUID': current_id, 'CASNUMBE': None}
            elif parts[1] == "CASNUMBE" and current_id:
                data['CASNUMBE'] = parts[2].replace("CAS ", "")
                if data['CASNUMBE']:  # Only append if CASNUMBE is not None or empty
                    ttd_data.append(data)

    unique_id = set()
    unique_cas = set()
    cas_to_ttdruid = {}

    for t in ttd_data:
        unique_id.add(t['TTDDRUID'])
        unique_cas.add(t['CASNUMBE'])
        if t['CASNUMBE'] in cas_to_ttdruid:
            cas_to_ttdruid[t['CASNUMBE']].append(t['TTDDRUID'])
        else:
            cas_to_ttdruid[t['CASNUMBE']] = [t['TTDDRUID']]
    
    # Count how many CAS numbers have multiple TTDDRUIDs
    cas_with_multiple_ttdruid = {cas: ttdruids for cas, ttdruids in cas_to_ttdruid.items() if len(ttdruids) > 1}

    print("CROSS-REF", len(unique_id), len(unique_cas))
    print("CAS numbers with multiple TTDDRUIDs:", len(cas_with_multiple_ttdruid))
    return ttd_data

def parse_file(input_file, ttd_cross_data):
    drugs = []
    current_drug = None
    unique = set()
    with open(input_file, 'r', encoding='utf-8') as file:
        lines = file.readlines()
        for line in tqdm(lines, desc="Parsing input file"):
            line = line.strip()
            if line.startswith("TTDDRUID"):
                if current_drug:
                    drugs.append(current_drug)
                current_drug = {
                    'TTDDRUID': line.split('\t')[1],
                    'drug_name': '',
                    'indications': []
                }
                found = False
                for t in ttd_cross_data:
                    if t['TTDDRUID'] == current_drug ['TTDDRUID']:
                        current_drug['CASNUMBE'] = t['CASNUMBE']
                        Found = True
                        break
                if not found:
                    current_drug['CASNUMBE'] = None
                unique.add(current_drug ['TTDDRUID'])
            elif line.startswith("DRUGNAME"):
                current_drug['drug_name'] = line.split('\t')[1]
            elif line.startswith("INDICATI"):
                parts = line.split('\t')
                if len(parts) >= 4:
                    icd_part = parts[2].split(': ')
                    icd_code = icd_part[1] if len(icd_part) > 1 else 'N.A.'
                    indication = {
                        'therapeutic_category': parts[1],
                        'ICD_11': icd_code,
                        'status': parts[3]
                    }
                    current_drug['indications'].append(indication)
                else:
                    print(f"Warning: Malformed line detected and skipped: {line}")

        if current_drug:
            drugs.append(current_drug)

    print("NUMBER OF ROWS", len(unique))
    return drugs

def load_drugbank_data(drugbank_file):
    with open(drugbank_file, 'rb') as file:
        return pickle.load(file)

def enrich_with_drugbank_data(drugs_data, drugbank_data):
    for drug in tqdm(drugs_data, desc="Enriching with DrugBank data"):
        found = False
        for db_entry in drugbank_data:
            if drug['drug_name'].lower() == db_entry['name'].lower():
                drug['drugbank_id'] = db_entry['drugbank_id']
                found = True
                break

        if not found:
            for db_entry in drugbank_data:
                if db_entry['products']:
                    for p in db_entry['products']:
                        if drug['drug_name'].lower() == p['name'].lower():
                            drug['drugbank_id'] = db_entry['drugbank_id']
                            found = True
                            break
                if found:
                    break
                
        if not found:
            for db_entry in drugbank_data:
                if drug['CASNUMBE'] == db_entry['cas_number']:
                    drug['drugbank_id'] = db_entry['drugbank_id']
                    found = True
                    break
    return drugs_data


def save_to_pickle(data, output_file):
    with open(output_file, 'wb') as file:
        pickle.dump(data, file)

def main():
    parser = argparse.ArgumentParser(description="Parse a drug data file, enrich with DrugBank data, and save as a pickle file.")
    parser.add_argument('input_file', type=str, help='Path to the input text file.')
    parser.add_argument('output_file', type=str, help='Path to the output pickle file.')
    parser.add_argument('cross_file', type=str, help='Path to the TTD cross-matching text file.')
    parser.add_argument('drugbank_file', type=str, help='Path to the DrugBank pickle file.')


    args = parser.parse_args()
    # Parse the TTD cross-matching file
    ttd_data = parse_ttd_cross_file(args.cross_file)
    # Create the cas_ prefixed filename
    output_dir = os.path.dirname(args.output_file)
    output_filename = os.path.basename(args.output_file)
    cas_output_file = os.path.join(output_dir, f"cas_{output_filename}")

    # Save the TTD data with CAS numbers
    save_to_pickle(ttd_data, cas_output_file)


    drugs_data = parse_file(args.input_file, ttd_data)
    if args.drugbank_file:
        u = set()
        drugbank_data = load_drugbank_data(args.drugbank_file)
        for d in drugbank_data:
            u.add(d['cas_number'])
        print("Number of unique cas_number", len(u))
        enriched_data = enrich_with_drugbank_data(drugs_data, drugbank_data)
        save_to_pickle(enriched_data, args.output_file)
        print(f"Data successfully saved to {args.output_file}")
        # Count total rows and rows without drugbank_id
        total_rows = len(enriched_data)
        rows_without_drugbank_id = sum(1 for drug in enriched_data if 'drugbank_id' not in drug)

        t = 0
        unique_drugs = set()
        for e in enriched_data:
            if 'drugbank_id' not in e and t < 10:
                print(e[''], e['drug_name'])
            t +=1
            unique_drugs.add(e['drugbank_id'])
        print(f"Total number of rows: {total_rows}")
        print(f"Total number of rows without a drugbank_id: {rows_without_drugbank_id}")
        print(f"Total unique drugs: ", len(unique_drugs))
if __name__ == "__main__":
    main()