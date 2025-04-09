# LLMDDI Online Repository
This is the online repository for the `LLMs for Drug-Drug Interaction Prediction` paper.

## Introduction
The repository is organized as follows:

| Directory                       | Description                                                                                                                |
| ------------------------------- | -------------------------------------------------------------------------------------------------------------------------- |
| [`external-datasets`]  | Contains the external datasets used for the empirical evaluation. |
| [`extractor`]  | Contains the source code used to extract data from the Drugbank dataset and convert the external datasets pickle files formats.              |
| [`feature_eng`] | Contains the source code used for the feature engineering process. |
| [`fine-tune`] | Contains the source code and configurations used to fine-tune LLMs. |
| [`model`] | Contains the replication code for the Suyu et al. l2 regression model. |
| [`MSDAFL`] | Contains the replication code for the Chao et al. MSDAFL deep learning model. |
| [`results`]   | Contains the empirical evaluation results. |


## External datasets
The `external-datasets` folder contains the external datasets from Serkan et al. used to validate the fine-tuned models.

Specifically, it contains the following datasets: 
- *Clinical knowledge bases*: CredibleMeds, HEP, and HIV.
- *Annotated corpora*: DDI Corpus 2011, DDI Corpus 2013, NLM Corpus, and PK DDI Corpus.
- *Healthcare systems*: OSCAR EMR and WorldVista.
- *Reference resources*: French DDI Referrals, KEGG, and NDF-RT.
- *Clinical guidelines*: ONC High Priority DDI List and ONC Non-Interuptive DDI List.

## Data Extraction
The `extractor` folder contains the python scripts to extract the necessary data used in our work.
In particular, it contains the following scripts:
- `pickle_extractor.py`, which extract data from the Drugbank dataset and create the the baseline dataset as pickle file.
- `extract_ddi.py`, which opens the original external datasets and convert their old pickle format to the binary pickle version.

To use the scripts, you need to proceed as follows:

### **SETUP**
### Step 1: Download the repository
Download the repository and go to the `LLMDDI` folder. 

### Step 2: Set up virtual environment and install packages
Install poetry.
Go to the `extractor` folder and run

```
poetry install 
```

You need to download the Drugbank xml file in order to launch the python scripts.


## Data Preprocessing and Feature Engineering
The `feature_eng` folder containts the python scripts to preprocess ddi data and create the training and validation sets used for our experiments.
Specifically, it contains the following scripts:
- `first_data_exploration.py`, which analyzes the drugbank pickle file created during the `Data Extraction` and print a few statistics.
- `get_missed_formulas.py`, uses external databases to gather the drugbank dataset missing information
- ddi:
    - `gs_ddi.py`, creates the compliant dataset, starting from the drugbank dataset, containing positive and negatives ddi instances
    - `gs_ddi_split.py`, splits the provided dataset in train and validation sets
    - `ext_data_set_norm.py`, reads the external datasets (the binary pickle versions), removes drugs which are not in drugbank and create the full ddi dataset, comprising all the external datasets ddi
    - gs_ddi_negs.py, creates a dataset containing only negative samples starting from the drugbank dataset.
    - `gs_ddi_ext_datasets_db_negs.py`, starts from the external datasets, and creates the dataset structure (compliant to our experiments) containing positives and negatives. The negative instances are extracted from an external pickle file, created from the drugbank dataset using gs_ddi_negs.py.
    - `gs_extract_train_val.py`, extracts training and validation samples from a dataset
    - `gs_ddi_embeddings.py`, updates the provided dataset computing embeddings for organism targeted by drugs and for SMILES formulas. We used MolFormer-XL for SMILES and text-embedding-3-small for organisms.
- `llm`:
    - `prep_gs_ddi_GPT_ft.py`, prepares the jsonl files for GPT-4 fine-tuning.
    - `prep_gs_ddi_Gemma_ft.py`, prepares the jsonl files for Gemma2 fine-tuning.
    - `prep_gs_ddi_DS_ft.py`, prepares the jsonl files for Deepseek r1 distilled models fine-tuning.
    - `fine_tune_GPT.py`, lauches the fine-tuning for GPT-4
    - `gs_ddi_GPT_eval.py`, evaluates the zero-shot/fine-tuning performance of the GPT-4 model on a specific dataset.
    - `gs_ddi_Gemini_eval.py`, evaluates the performance of the Gemini model on a specific dataset.
    - `gs_ddi_Open_LLM_eval.py`, evaluates the zero-shot/fine-tuning performance of Open LLM models on a specific dataset.
    - `gs_ddi_Claude_eval.py`, evaluates the performance of the Claude model on a specific dataset.
    - `analyze_gs_ddi_LLM.py`, compute metrics on the evaluation performed using eval scripts.
    - `fs`, which contains the few-shots scripts:
        - `gs_ddi_GPT_eval.py`, evaluates the few-shot performance of the GPT-4 model on a specific dataset.
        - `gs_ddi_Gemini_eval.py`, evaluates the few-shot performance of the Gemini model on a specific dataset.
        - `gs_ddi_Open_LLM_eval.py`, evaluates the few-shot performance of Open LLM models on a specific dataset.
        - `gs_ddi_Claude_eval.py`, evaluates the few-shot performance of the Claude model on a specific dataset.

To use the scripts, you need to proceed as follows:

### **SETUP**
### Step 1: Download the repository
Download the repository and go to the `LLMDDI` folder. 

### Step 2: Set up virtual environment and install packages
Install poetry.
Go to the `feature_eng` folder and run

```
poetry install 
```

You need an account with OpenAI, Google, and Anthropic in order to use their models. You also need to provide the necessary API keys.

## Fine-tuning
The `fine-tune` folder contains the scripts and configurations files to find out the optimized hyperparameters and to fine-fune LLMs using LoRA.
In particular, it contains the following scripts and files:
- `setup_env.sh`, which setup the necessary python libraries and virtual envs.
- `best_config.py`, which finds out the best hyperparameter configurations for a specific LLM
- `fine_tune.sh`, which starts fine-tuning a specific LLM.
- `optuna_config.yaml`, contains the Optuna configuration for the optimization step.
- _LLM Model_`-config`_epochs_`.yaml`, ontains the LoRA configuration for the _LLM Model_ and the number of _epochs_

## L2 Model replication
The `model` folder contains the source code to train and evaluate the L2 model proposed by Suyu et al.
Specifically, it contains the following scripts:
- `L2_model.py`, which trains the L2 model
- `eval_L2_model.py`, which evaluates the L2 model on the validation datasets (also external ones.)

To use the scripts, you need to proceed as follows:

### **SETUP**
### Step 1: Download the repository
Download the repository and go to the `LLMDDI` folder. 

### Step 2: Set up virtual environment and install packages
Install poetry.
Go to the `model` folder and run

```
poetry install 
```

## MSDAFL Model replication
The `MSDAFL` folder contains the source code to create the datasets, train, and evaluate the MSDAFL model proposed by Chao et al.
Specifically, it contains the following scripts:
- `data_prep`: 
    - `1_create_dict.py` which creates the drug dictionary for the datsets in a given folder 
    - `2_create_smiles.py` which creates the JSON file containing the smiles of the drug dictionary create with the `1_create_dict.py` script
    - `3_prepare_graph.py` which create the graph for the smiles created with the `2_create_smiles.py` script
    - `4_create_datasets.py` which create the JSON version required by MSDAFL to train, validate, and test the model
- `model`, which contains the original codes developed by Chao et al. and the `ddi_train_msdafl.py` script needed to train, validate, and test the MSDAFL model on the datasets (also external ones) of our study.

To use the scripts, you need to proceed as follows:

### **SETUP**
### Step 1: Download the repository
Download the repository and go to the `MSDAFL` folder. 

### Step 2: Set up virtual environment and install packages
Install poetry.
Go to the `msdafl` folder and run

```
poetry install 
```


## Results
The `results` folder contains the results for the the experiments.
In particular, it contains the following:
- `fine-tuning`, contains the results for the LLM fine-tuned versions.
- `L2`, contains the results for the L2 model.
- `MSDAFL`, contains the results for the MSDAFL model.
- `zero-shot`, contains the results for the zero-shot experiment with LLMs.
- `few-shot`, contains the results for the few-shot experiments with LLMs.

