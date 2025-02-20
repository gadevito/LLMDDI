#
# Update the provided dataset computing embeddings for organism targeted by drugs and
# for SMILES formulas. We used MolFormer-XL for SMILES and text-embedding-3-small for organisms
#
from openai import OpenAI
from transformers import AutoModelForMaskedLM, AutoTokenizer
import torch
import os
import pickle
from tqdm import tqdm
import argparse
import numpy as np
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
import warnings
from transformers import logging as transformers_logging

transformers_logging.set_verbosity_error()
warnings.filterwarnings("ignore", category=UserWarning, module="transformers")
warnings.simplefilter("ignore", UserWarning)

os.environ["TOKENIZERS_PARALLELISM"] = "false"

os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
os.environ['HF_HOME'] = 'MODELS_CACHE'
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")


chemberta = AutoModelForMaskedLM.from_pretrained("DeepChem/ChemBERTa-77M-MTR")
chemberta_tokenizer = AutoTokenizer.from_pretrained("DeepChem/ChemBERTa-77M-MTR")
chemberta.to(device)


molformer = AutoModelForMaskedLM.from_pretrained("ibm/MoLFormer-XL-both-10pct", trust_remote_code=True, output_hidden_states=True)
molformer_tokenizer = AutoTokenizer.from_pretrained("ibm/MoLFormer-XL-both-10pct", trust_remote_code=True)
molformer.to(torch.device("cpu"))


smolebert = AutoModelForMaskedLM.from_pretrained("UdS-LSV/smole-bert", output_hidden_states=True)
smolebert_tokenizer = AutoTokenizer.from_pretrained("UdS-LSV/smole-bert")
smolebert.to(device)


simcsesmolebert = AutoModelForMaskedLM.from_pretrained("UdS-LSV/muv2x-simcse-smole-bert", output_hidden_states=True)
simcsesmolebert_tokenizer = AutoTokenizer.from_pretrained("UdS-LSV/muv2x-simcse-smole-bert")
simcsesmolebert.to(device)

siamesesmolebert = AutoModelForMaskedLM.from_pretrained("UdS-LSV/siamese-smole-bert-muv-1x", output_hidden_states=True)
siamesesmolebert_tokenizer = AutoTokenizer.from_pretrained("UdS-LSV/siamese-smole-bert-muv-1x")
siamesesmolebert.to(device)

client = OpenAI()

gp_embeddings_cache ={}

def loadPickle(input_pickle):
    with open(input_pickle, 'rb') as f:
        r = pickle.load(f)
    return r

# Available models
# text-embedding-3-large --> 3,072
# text-embedding-3-small --> 1,536
# text-embedding-ada-002 --> 1,536
def get_gp_embeddings(text, model="text-embedding-3-small"):
    if not text or len(text.strip()) ==0:
        return [0.0] * (3072 if model == "text-embedding-3-large" else 1536)
    for attempt in range(5):
        try:
            if text in gp_embeddings_cache:
                return gp_embeddings_cache.get(text)
            response = client.embeddings.create(
                            model=model,
                            input=text,
                            encoding_format="float"
                        )
            gp_embeddings_cache[text] = response.data[0].embedding
            return response.data[0].embedding
        except Exception as e:
                print("An error occurred during processing. Saving current progress...")
                print(traceback.format_exc())
                wait = 2 * (2 ** attempt)
                time.sleep(wait)
    raise Exception(f"Max retries exceeded {model}")


def get_smiles_embeddings(smiles, model="ChemBERTa"):
    padding = True

    max_length = 512  # Puoi adattare questo valore in base alle tue necessità
    embedding_size=512
    if model == "ChemBERTa":
        embedding_size= 600
        if not smiles:
            return np.zeros(embedding_size)
        encoded_input = chemberta_tokenizer(smiles, return_tensors="pt",padding=padding,truncation=True, max_length=max_length)
        encoded_input = {key: value.to(device) for key, value in encoded_input.items()}
        with torch.no_grad():
            model_output = chemberta(**encoded_input)
        
        # Estrai l'embedding dalla prima posizione e spostalo sulla CPU prima di convertire in NumPy
        embedding = model_output[0][:, 0, :].detach().cpu().numpy()
        
        return embedding.squeeze()
    elif model == "MolFormer-XL":
        embedding_size= 768
        if not smiles:
            return np.zeros(embedding_size)
        encoded_input = molformer_tokenizer(smiles, return_tensors="pt",padding=padding,truncation=True, max_length=max_length)
        #encoded_input = {key: value.to(device) for key, value in encoded_input.items()}
        encoded_input = {key: value.to('cpu') for key, value in encoded_input.items()}
        with torch.no_grad():
            model_output = molformer(**encoded_input)
            hidden_states = model_output.hidden_states

        # Estrai l'embedding dal token [CLS]
        cls_embedding = hidden_states[-1][:, 0, :].detach().cpu().numpy()
        return cls_embedding.squeeze()        

    elif model == "SMoleBert":
        if not smiles:
            return np.zeros(embedding_size)
        encoded_input = smolebert_tokenizer(smiles, return_tensors="pt",padding=padding,truncation=True, max_length=max_length)
        encoded_input = {key: value.to(device) for key, value in encoded_input.items()}
        #encoded_input = {key: value.to('cpu') for key, value in encoded_input.items()}
        with torch.no_grad():
            model_output = smolebert(**encoded_input)
            hidden_states = model_output.hidden_states

        # Estrai l'embedding dal token [CLS]
        cls_embedding = hidden_states[-1][:, 0, :].detach().cpu().numpy()
        return cls_embedding.squeeze()

    elif model == "SimcseSmoleBert": 
        if not smiles:
            return np.zeros(embedding_size)
        encoded_input = simcsesmolebert_tokenizer(smiles, return_tensors="pt",padding=padding,truncation=True, max_length=max_length)
        encoded_input = {key: value.to(device) for key, value in encoded_input.items()}
        #encoded_input = {key: value.to('cpu') for key, value in encoded_input.items()}
        with torch.no_grad():
            model_output = simcsesmolebert(**encoded_input)
            hidden_states = model_output.hidden_states

        # Estrai l'embedding dal token [CLS]
        cls_embedding = hidden_states[-1][:, 0, :].detach().cpu().numpy()
        return cls_embedding.squeeze()

    elif model == "SiameseSmoleBert": 
        if not smiles:
            return np.zeros(embedding_size)
        encoded_input = siamesesmolebert_tokenizer(smiles, return_tensors="pt",padding=padding,truncation=True, max_length=max_length)
        encoded_input = {key: value.to(device) for key, value in encoded_input.items()}
        #encoded_input = {key: value.to('cpu') for key, value in encoded_input.items()}
        with torch.no_grad():
            model_output = siamesesmolebert(**encoded_input)
            hidden_states = model_output.hidden_states

        # Estrai l'embedding dal token [CLS]
        cls_embedding = hidden_states[-1][:, 0, :].detach().cpu().numpy()
        return cls_embedding.squeeze()

    return None

def process_line(t, model):
    smiles1 = t[2]
    smiles2 = t[3]
    drug1_organism = t[4]
    drug2_organism = t[5]

    if smiles1:
        if isinstance(smiles1, float):
            smiles1 = ""
        smiles1 = smiles1.strip()
    smiles1_emb = get_smiles_embeddings(smiles1,model)

    if smiles2:
        if isinstance(smiles2, float):
            smiles2 = ""
        smiles2 = smiles2.strip()
    smiles2_emb = get_smiles_embeddings(smiles2,model)

    org1 = get_gp_embeddings(drug1_organism)
    org2 = get_gp_embeddings(drug2_organism)

    new_t = (
                t[0],  # drug1_id
                t[1],  # drug2_id
                *smiles1_emb,        # embeddings for smiles1
                *smiles2_emb,        # embeddings for smiles2
                t[6],  # enc_drug1_organism
                t[7],  # enc_drug2_organism
                *org1,               # embeddings for org1
                *org2,               # embeddings for org2
                *t[8:]  # rest of the original tuple after enc_drug2_organism
    )
    return new_t


def parallel_process_dataset(file_name, output_pickle, model):
    try:
        drugs = loadPickle(file_name)
    except Exception as e:
        return "NOK"
    bn = os.path.basename(file_name)

    # Usa ProcessPoolExecutor per il parallelismo
    with ProcessPoolExecutor() as executor:
        # Mappa le righe per la funzione di elaborazione parallela
        futures = {executor.submit(process_line, drugs[i], model): i for i in range(len(drugs))}

        # Inizializza la barra al di fuori del ciclo principale
        with tqdm(total=len(futures), desc=f"Processing {bn}") as pbar:
            # Raccoglie e itera sui futuri man mano che si completano
            for future in as_completed(futures):
                index = futures[future]
                try:
                    drugs[index] = future.result()
                except Exception as e:
                    print(f"Errore processo alla riga {index}: {e}")
                pbar.update(1)  # Aggiorna la barra ad ogni futuro completato

    # Salva i risultati finali
    with open(output_pickle, 'wb') as f:
        pickle.dump(drugs, f)

    print("Dataset updated!")
    return "OK"

def process_dataset(file_name, output_pickle, model):
    try:
        drugs = loadPickle(file_name)
    except Exception as e:
        return "NOK"
    bn = os.path.basename(file_name)
    for i in tqdm(range(len(drugs)), desc=f"Updating smiles and organisms: {bn}"):
        t = drugs[i]
        smiles1 = t[2]
        smiles2 = t[3]
        drug1_organism = t[4]
        drug2_organism = t[5]
        try:
            if smiles1:
                if isinstance(smiles1, float):
                    smiles1 = ""
                smiles1 = smiles1.strip()
            smiles1_emb = get_smiles_embeddings(smiles1,model)
        except Exception as e:
            print(f"Tipo di smiles: {type(smiles1)}\nValore di smiles: {repr(smiles1)}")
            print(smiles1)
            raise e
        try:
            if smiles2:
                if isinstance(smiles2, float):
                    smiles2 = ""
                smiles2 = smiles2.strip()
            smiles2_emb = get_smiles_embeddings(smiles2,model)
        except Exception as e:
            print(f"Tipo di smiles: {type(smiles2)}\nValore di smiles: {repr(smiles2)}")
            print(smiles2)
            raise e
        try:
            org1 = get_gp_embeddings(drug1_organism)
        except Exception as e:
            print(f"Tipo di smiles: {type(drug1_organism)}\nValore di organism1: {repr(drug1_organism)}")
            print(drug1_organism)
            raise e
        try:
            org2 = get_gp_embeddings(drug2_organism)
        except Exception as e:
            print(f"Tipo di smiles: {type(drug2_organism)}\nValore di organism2: {repr(drug2_organism)}")
            print(drug2_organism)
            raise e
        try:
            new_t = (
                t[0],  # drug1_id
                t[1],  # drug2_id
                *smiles1_emb,        # embeddings for smiles1
                *smiles2_emb,        # embeddings for smiles2
                t[6],  # enc_drug1_organism
                t[7],  # enc_drug2_organism
                *org1,               # embeddings for org1
                *org2,               # embeddings for org2
                *t[8:]  # Il resto della tupla dopo enc_drug2_organism
            )

            drugs[i] = new_t
        except Exception as e:
            print(f"smiles1_emb: {type(smiles1_emb)} - {smiles1_emb}")
            print(f"smiles2_emb: {type(smiles2_emb)} - {smiles2_emb}")
            print(f"org1: {type(org1)} - {org1}")
            print(f"org2: {type(org2)} - {org2}")
            raise e
    with open(output_pickle, 'wb') as f:
        pickle.dump(drugs, f)

    print("Dataset updated!")
    return "OK"

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Process all the dataset in a specific folder to add embeddings for organisms and SMILES formulas.")
    parser.add_argument('input_pickle', type=str, help='Pickle path containing the datasets.')
    parser.add_argument('output_pickle', type=str, help='Pickle path to save the processed datasets.')
    parser.add_argument('model', type=str, help='Model to use to compute SMILES embeddings: ChemBERTa, MolFormer-XL, SMoleBert, SimcseSmoleBert, SiameseSmoleBert.')
    parser.add_argument('--parallel', action='store_true', help='Parallel execution')
    
    args = parser.parse_args()

    ds_path = args.input_pickle

    files = os.listdir(ds_path)
    full_file_paths = [os.path.join(ds_path, f) for f in files if os.path.isfile(os.path.join(ds_path, f))]
    cont = 1
    tot = len(files)
    for f in tqdm(full_file_paths, desc="Processing datasets"):
        #if f == "../extractor/extracted_data/ddi_datasets/samples/gs_dataset.pkl":
        #    continue
        print(f"Processing {cont}/{tot}. Dataset {f}")
        ofile_name = os.path.basename(f)
        output_file_name = os.path.join(args.output_pickle,ofile_name)
        if args.parallel:
            parallel_process_dataset(f, output_file_name, args.model)
        else:
            process_dataset(f, output_file_name, args.model)
