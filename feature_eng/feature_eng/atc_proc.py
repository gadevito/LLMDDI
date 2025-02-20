import csv
import pickle
import argparse
from collections import defaultdict
from sklearn.preprocessing import MultiLabelBinarizer
from typing import List

class ATCProcessor:
    def __init__(self, file):
        self.file = file
        self.all_codes = set()
        self.parent_map = defaultdict(set)
        self._selected_codes = set()
        self.mlb = None
        if self.file.endswith('.pkl'):
            with open(self.file, 'rb') as file:
                self._selected_codes = pickle.load(file)

    @property
    def selected_codes(self):
        return self._selected_codes

    def createBinaryRepresentation(self):
        unique_atc_codes = list(self._selected_codes)
        self.mlb = MultiLabelBinarizer(classes=unique_atc_codes)
        self.mlb.fit([unique_atc_codes])
    
    def transform_single_drug_atc(self, drug_atc_codes:List):        
        # Trasforma i livelli in un vettore binario
        return self.mlb.transform([drug_atc_codes])[0]

    def process_atc_codes(self):

        with open(self.file, mode='r', encoding='utf-8') as file:
            reader = csv.DictReader(file)
            for row in reader:
                atc_code = row['atc_code']
                if atc_code[0:1] not in ('A', 'B', 'C', 'D', 'G', 'H', 'J', 'L', 'M', 'N', 'P', 'R', 'S', 'V'):
                    continue
                self.all_codes.add(atc_code)
                
        # Seleziona i codici di quinto livello e quelli senza figli
        for code in self.all_codes:
            if len(code) == 7 or not any(child.startswith(code) and child != code for child in self.all_codes):
                self._selected_codes.add(code)

    def save_to_pickle(self, pickle_file):
        with open(pickle_file, 'wb') as file:
            pickle.dump(self._selected_codes, file)
        
        print(len(self._selected_codes))





if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Estrazione dei dati sui farmaci da un file XML.')
    parser.add_argument('--atc_file', type=str, required=False,  default='../WHO ATC-DDD 2024-07-31.csv', help='CSV file containing ATC codes')
    parser.add_argument('--pickle_file', type=str, required=True, default='atc.pkl', help='Pickle file name for the processed ATC codes')
    
    args = parser.parse_args()

    csv_file = args.atc_file # 'atc_codes.csv'  # Il tuo file CSV
    pickle_file = args.pickle_file #'top_level_atc_codes.pkl'

    processor = ATCProcessor(csv_file)
    processor.process_atc_codes()
    processor.save_to_pickle(pickle_file)

    processor = ATCProcessor(pickle_file)
    processor.createBinaryRepresentation()

    binary = processor.transform_single_drug_atc(['A01AA02', 'A01AA03'])
    print(binary)



