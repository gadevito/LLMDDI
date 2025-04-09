import pickle
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import traceback
from openai import OpenAI
import time
from tqdm import tqdm
import argparse

#
# Calculate the embeddings for the textual information: SMILES, organisms and genes.
# Find similar examples in the train set based on SMILES, organisms and genes.
# It can also find random examples in the training set.
#
class SimilaritySearchSelector:
    def __init__(self, drugbank_pickle, dataset_path, processed_dataset_path, model="text-embedding-3-small"):
        """
        Initialize the class, loading the drugbank dataset, the training dataset and/or the preprocessed embeddings for the training set.
        
        Args:
            drugbank_pickle (str): drugbank pickle path
            dataset_path (str): training set pickle path 
            processed_dataset_path (str): pickle file of training set embeddings
            model (str): embedding model from OpenAI
        """

        self.model = model
        # We first load the drugbank file
        self.drugbank_pickle = drugbank_pickle
        self.dataset = self.load_pickle(dataset_path)
        # Then, we preprocess the training set file
        self.preproc_dataset()

        found = False
        # Check if the embeddings have been already computed
        try:
            all_data = self.load_pickle(processed_dataset_path)
            self.smiles1_embeddings = all_data['smiles1_embeddings']
            self.smiles2_embeddings = all_data['smiles2_embeddings']
            self.org1_embeddings = all_data['org1_embeddings']
            self.org2_embeddings = all_data['org2_embeddings']
            self.genes1_embeddings = all_data['genes1_embeddings']
            self.genes2_embeddings = all_data['genes2_embeddings']
            found = True
            print("EMBEDDINGS found.....")
        except Exception as e:
            pass
        
        self.gp_embeddings_cache ={}
        self.client = OpenAI()
        # If embeddings have not been computed yet, then we calculate them
        if not found:

            print("Calculate EMBEDDINGS....")
            # Compute embeddings and save the results in a pickle file for future loading
            self._precompute_embeddings()
            self.save_pickle(processed_dataset_path)


    # Get the list of genes targeted by the drug
    def get_human_targets(self, drug):
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

    # Check if the drug has been approved or experimental, but not illicit or withdrawn
    def has_approved_group(self, d):
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

    # Get organisms
    def get_organisms(self,drugs):
        drug_org_dict = {} 
        for drug in drugs:
            drug_id = drug['drugbank_id']
            drug_org_dict[drug_id] = ''
            if 'targets' in drug:
                for target in drug['targets']:
                    drug_org_dict[drug_id] = target['organism']
                

        return drug_org_dict

    # Preprocess the training set file
    def preproc_dataset(self):
        drugs = self.load_pickle(self.drugbank_pickle)
        for drug in drugs:
            smile = drug.get('calc_prop_smiles','')
            if not smile or isinstance(smile, float):
                smile = ''
            drug['calc_prop_smiles'] = smile

        # First, we filter data that belongs to the right groups
        drugs = [
            {key: drug[key] for key in ['drugbank_id', 'name', 'targets', 'calc_prop_smiles', 'drug_interactions'] if key in drug}
            for drug in drugs if self.has_approved_group(drug)
        ]

        all_human_genes = set()
        for drug in drugs:
            human_targets = self.get_human_targets(drug)
            all_human_genes.update(human_targets)

        # Sort genes for consistency
        all_human_genes = sorted(all_human_genes)

        # Remove drugs that do not target genes
        drugs = [drug for drug in drugs if self.get_human_targets(drug)]

        drug_org_dict = self.get_organisms(drugs)
        # Create a dictionary to access drugs that target genes given their ids
        drug_dict = {drug['drugbank_id']: drug for drug in drugs}

        res = []

        for i in range(len(self.dataset)):
            t = self.dataset[i]
            drug1 = t[0]
            drug2 = t[1]

            cdrug1 = drug_dict[drug1]
            cdrug2 = drug_dict[drug2]
            res.append({
                "drug1": drug1,
                "drug2": drug2,
                "drug_name1": cdrug1['name'],
                "drug_name2": cdrug2['name'],
                "smiles1": cdrug1['calc_prop_smiles'],
                "smiles2": cdrug2['calc_prop_smiles'],
                "genes1": self.get_human_targets(cdrug1),
                "genes2": self.get_human_targets(cdrug2),
                "org1": drug_org_dict[drug1],
                "org2": drug_org_dict[drug2],
                "target": t[-1]
            })

        self.dataset = res

    # Save the precomputed embeddings
    def save_pickle(self, dataset_path):
        drugs = {"smiles1_embeddings":self.smiles1_embeddings,
                 "smiles2_embeddings":self.smiles1_embeddings,
                 "org1_embeddings": self.org1_embeddings,
                 "org2_embeddings": self.org2_embeddings,
                 "genes1_embeddings": self.genes1_embeddings,
                 "genes2_embeddings": self.genes2_embeddings}
        with open(dataset_path, 'wb') as f:
            pickle.dump(drugs, f)

    def load_pickle(self, dataset_path):
        # Carica il dataset
        dataset = None
        with open(dataset_path, 'rb') as f:
            dataset = pickle.load(f)
        return dataset
    
    # Available models
    # text-embedding-3-large --> 3,072
    # text-embedding-3-small --> 1,536
    # text-embedding-ada-002 --> 1,536
    def get_gp_embeddings(self, text):
        if not text or len(text.strip()) ==0:
            return [0.0] * (3072 if self.model == "text-embedding-3-large" else 1536)
        for attempt in range(5):
            try:
                if text in self.gp_embeddings_cache:
                    return self.gp_embeddings_cache.get(text)
                response = self.client.embeddings.create(
                                model=self.model,
                                input=text,
                                encoding_format="float"
                            )
                self.gp_embeddings_cache[text] = response.data[0].embedding
                return response.data[0].embedding
            except Exception as e:
                    print("An error occurred during processing. Saving current progress...")
                    print(traceback.format_exc())
                    wait = 2 * (2 ** attempt)
                    time.sleep(wait)
        raise Exception(f"Max retries exceeded {self.model}")



    # Calculate embeddings
    def _precompute_embeddings(self):
        # Extract features from the training set 
        smiles1_list = [item['smiles1'] for item in self.dataset]
        smiles2_list = [item['smiles2'] for item in self.dataset]
        org1_list = [item['org1'] for item in self.dataset]
        org2_list = [item['org2'] for item in self.dataset]
        genes1_list = [item['genes1'] for item in self.dataset]
        genes2_list = [item['genes2'] for item in self.dataset]
        
        # Calculate embeddings for each feature 
        print("SMILES1....")
        self.smiles1_embeddings = [self.get_gp_embeddings(item) for item in tqdm(smiles1_list)]
        print("SMILES2....")
        self.smiles2_embeddings = [self.get_gp_embeddings(item) for item in tqdm(smiles2_list)]
        print("ORG1....")
        self.org1_embeddings = [self.get_gp_embeddings(item) for item in tqdm(org1_list)]
        print("ORG2....")
        self.org2_embeddings = [self.get_gp_embeddings(item) for item in tqdm(org2_list)]
        print("GENES1....")
        self.genes1_embeddings = [self.get_gp_embeddings("|".join(item)) for item in tqdm(genes1_list)]
        print("GENES2....")
        self.genes2_embeddings = [self.get_gp_embeddings("|".join(item)) for item in tqdm(genes2_list)]
    
    def get_random_balanced_entries(self, num_results=10):
        """
        Select random entries from the dataset, balanced between target=1 and target=0.
        
        Args:
            num_results (int): Number of entries to return
        
        Returns:
            list: List of randomly selected entries with balanced targets
        """
        # Create a DataFrame from the dataset for easier filtering
        df = pd.DataFrame({
            'index': range(len(self.dataset)),
            'target': [item['target'] for item in self.dataset]
        })
        
        # Calculate how many examples of each class we need
        half_size = num_results // 2
        remainder = num_results % 2  # In case num_results is odd
        
        # Filter entries by target
        positive_indices = df[df['target'] == 1]['index'].values
        negative_indices = df[df['target'] == 0]['index'].values
        
        # Check if we have enough samples in each class
        if len(positive_indices) < half_size or len(negative_indices) < half_size + remainder:
            # If not enough samples, adjust the number of samples per class
            pos_samples = min(half_size, len(positive_indices))
            neg_samples = min(half_size + remainder, len(negative_indices))
            
            # If one class doesn't have enough samples, take more from the other
            if pos_samples < half_size:
                neg_samples = min(num_results - pos_samples, len(negative_indices))
            if neg_samples < half_size + remainder:
                pos_samples = min(num_results - neg_samples, len(positive_indices))
        else:
            pos_samples = half_size
            neg_samples = half_size + remainder
        
        # Random sampling
        sampled_positive = np.random.choice(positive_indices, pos_samples, replace=False)
        sampled_negative = np.random.choice(negative_indices, neg_samples, replace=False)
        
        # Combine the selected indices
        result_indices = np.concatenate([sampled_positive, sampled_negative])
        
        # Shuffle the results to mix positive and negative examples
        np.random.shuffle(result_indices)
        
        # Return the randomly selected entries
        return [self.dataset[idx] for idx in result_indices]

    # Find the top num_results entries fimilar to the given case
    def find_similar_entries(self, query_dict, num_results=10, rand=False):
        """
        Look for similar entries in the training set using embeddings and cosine similarity.
        
        Args:
            query_dict (dict): New case for which we want to look for similar entries 
            num_results (int): Number of similar entries to look for
        
        Returns:
            list: List of similar entries 
        """
        if rand:
            return self.get_random_balanced_entries(num_results)
        
        # Extract features for the new case 
        n_smiles1 = query_dict["smiles1"]
        n_smiles2 = query_dict["smiles2"]
        n_org1 = query_dict["org1"]
        n_org2 = query_dict["org2"]
        n_genes1 = "|".join(query_dict["genes1"])  # Convert the gene list in a single string where genes are separated by "|"
        n_genes2 = "|".join(query_dict["genes2"])  # Convert the gene list in a single string where genes are separated by "|"
        
        # Compute embeddings for the given case 
        n_smiles1_emb = self.get_gp_embeddings(n_smiles1)
        n_smiles2_emb = self.get_gp_embeddings(n_smiles2)
        n_org1_emb = self.get_gp_embeddings(n_org1)
        n_org2_emb = self.get_gp_embeddings(n_org2)
        n_genes1_emb = self.get_gp_embeddings(n_genes1)
        n_genes2_emb = self.get_gp_embeddings(n_genes2)
        
        # Calculate similarities for each feature 
        sim_smiles1 = cosine_similarity([n_smiles1_emb], self.smiles1_embeddings)[0]
        sim_smiles2 = cosine_similarity([n_smiles2_emb], self.smiles2_embeddings)[0]
        sim_org1 = cosine_similarity([n_org1_emb], self.org1_embeddings)[0]
        sim_org2 = cosine_similarity([n_org2_emb], self.org2_embeddings)[0]
        sim_genes1 = cosine_similarity([n_genes1_emb], self.genes1_embeddings)[0]
        sim_genes2 = cosine_similarity([n_genes2_emb], self.genes2_embeddings)[0]
        
        # Calculate the weighted sum of similarities 
        similarity_scores = (sim_smiles1 + sim_smiles2 + sim_org1 + sim_org2 + sim_genes1 + sim_genes2) / 6
        
        # Create a DataFrame to make easier the results selection 
        df = pd.DataFrame({
            'index': range(len(self.dataset)),
            'score': similarity_scores,
            'target': [item['target'] for item in self.dataset]
        })
        
        # Try to balance the results 50/50 between target=0 and target=1
        half_size = num_results // 2
        
        # Select top half_size with target=1
        top_positive = df[df['target'] == 1].sort_values('score', ascending=False).head(half_size)
        
        # Select top half_size with target=0
        top_negative = df[df['target'] == 0].sort_values('score', ascending=False).head(half_size)
        
        # Join the results 
        result_indices = pd.concat([top_positive, top_negative])['index'].values
        
        # Return the similar entries 
        return [self.dataset[idx] for idx in result_indices]
    
    # Format a list of examples
    def format_examples(self,examples):
        res = [self.example_formatter(f, idx+1) for idx, f in enumerate(examples)]

        result = "\n\n".join(res)
        return "** Examples **\n\n" + result

    # Format a single example
    def example_formatter(self, example, idx):

        template = """{idx}. Drug1: {drug1}
   SMILES for drug1: {smiles1}
   Organism targeted by drug1: {org1}
   Genes targeted by drug1: {genes1}

   Drug2: {drug2}
   SMILES for drug2: {smiles2}
   Organism targeted by drug2: {org2}
   Genes targeted by drug2: {genes2}

   CLASSIFICATION: {classification}"""

        formatted_example = template.format(idx=idx,
                                            drug1=example["drug_name1"],
                                            smiles1=example["smiles1"],
                                            org1=example["org1"],
                                            genes1=", ".join(example["genes1"]),
                                            drug2=example["drug_name2"],
                                            smiles2=example["smiles2"],
                                            org2=example["org2"],
                                            genes2=", ".join(example["genes2"]),
                                            classification="interaction" if example["target"] == 1 else "no interaction"
                                            )
        return formatted_example

    # Return the detailed similarity results
    def get_detailed_similarity(self, query_dict, entry_idx):
        """
        Compute and return the detailed similarities for each feature.
        
        Args:
            query_dict (dict): New case used to look for similar entries 
            entry_idx (int): Element index in the dataset
        
        Returns:
            dict: Dictionaty with the detailed similarities 
        """
        # Extract features from the new case 
        n_smiles1 = query_dict["smiles1"]
        n_smiles2 = query_dict["smiles2"]
        n_org1 = query_dict["org1"]
        n_org2 = query_dict["org2"]
        n_genes1 = "|".join(query_dict["genes1"])
        n_genes2 = "|".join(query_dict["genes2"])
        
        # Compute embedding for the new case 
        n_smiles1_emb = self.get_gp_embeddings(n_smiles1)
        n_smiles2_emb = self.get_gp_embeddings(n_smiles2)
        n_org1_emb = self.get_gp_embeddings(n_org1)
        n_org2_emb = self.get_gp_embeddings(n_org2)
        n_genes1_emb = self.get_gp_embeddings(n_genes1)
        n_genes2_emb = self.get_gp_embeddings(n_genes2)
        
        # Compute similarities for each feature 
        similarity = {
            'smiles1': float(cosine_similarity([n_smiles1_emb], [self.smiles1_embeddings[entry_idx]])[0][0]),
            'smiles2': float(cosine_similarity([n_smiles2_emb], [self.smiles2_embeddings[entry_idx]])[0][0]),
            'org1': float(cosine_similarity([n_org1_emb], [self.org1_embeddings[entry_idx]])[0][0]),
            'org2': float(cosine_similarity([n_org2_emb], [self.org2_embeddings[entry_idx]])[0][0]),
            'genes1': float(cosine_similarity([n_genes1_emb], [self.genes1_embeddings[entry_idx]])[0][0]),
            'genes2': float(cosine_similarity([n_genes2_emb], [self.genes2_embeddings[entry_idx]])[0][0]),
            'overall': float(np.mean([
                cosine_similarity([n_smiles1_emb], [self.smiles1_embeddings[entry_idx]])[0][0],
                cosine_similarity([n_smiles2_emb], [self.smiles2_embeddings[entry_idx]])[0][0],
                cosine_similarity([n_org1_emb], [self.org1_embeddings[entry_idx]])[0][0],
                cosine_similarity([n_org2_emb], [self.org2_embeddings[entry_idx]])[0][0],
                cosine_similarity([n_genes1_emb], [self.genes1_embeddings[entry_idx]])[0][0],
                cosine_similarity([n_genes2_emb], [self.genes2_embeddings[entry_idx]])[0][0]
            ]))
        }
        
        return similarity
    

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Process the dataset and create embeddings") 
    parser.add_argument('drugbank_pickle', type=str, help='Pickle path containing the drugbank dataset.')
    parser.add_argument('input_pickle', type=str, help='Pickle path containing the dataset.')
    parser.add_argument('output_pickle', type=str, help='Pickle path to save the processed datasets.')
    parser.add_argument('model', type=str, help='Model to use to compute embeddings: text-embedding-3-large, text-embedding-3-small, text-embedding-ada-002.')

    # text-embedding-3-large --> 3,072
    # text-embedding-3-small --> 1,536
    # text-embedding-ada-002 --> 1,536
    args = parser.parse_args()

    sel = SimilaritySearchSelector(args.drugbank_pickle, args.input_pickle, args.output_pickle)


    drugs = {"smiles1":"[H][C@]12SCC(COC(C)=O)=C(N1C(=O)[C@H]2NC(=O)C(=N/OC)\\C1=CSC(N)=N1)C(O)=O",
             "org1":"Humans",
             "genes1":["pbp1b", "pbp2a", "pbpC", "pbpA", "penA", "SLC22A6", "SLC22A8", "SLC22A11", "SLC22A7", "SLC15A1", "ALB", "SLC15A2"],
             "smiles2":"[H][C@]12SC(C)(C)[C@@H](N1C(=O)[C@H]2NC(=O)CC1=CC=CC=C1)C(O)=O",
             "org2": "Humans",
             "genes2":["pbp3", "SLC22A8", "SLC15A1", "SLC15A2"]
             }

    res = sel.find_similar_entries(drugs)



    print(sel.format_examples(res))