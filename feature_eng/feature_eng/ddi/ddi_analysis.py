#
# Analyze the DDI in the specific drug dataset
#
import pickle
import argparse
from collections import defaultdict

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

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Analyze DDI.')
    parser.add_argument('pickle_file', type=str, help='Drugbank dataset.')
    args = parser.parse_args()
    main(args.pickle_file)