## EXTRACTED DATASETS ##
This folder contains the dataset processed for the experiments:
- compliant_datasets, contains the external datasets converted to a compliance form
- full-known-interactions, contains the full ddi reported in all datasets, drugbank and external datasets
- samples, contains the dataset samples extracted from the drugbank dataset. It also contains the following folders:
    - 1k, contains 1000 samples extracted from the drugbank dataset used for training the models
    - 20k, contains 20000 samples extracted from the drugbank dataset used for training the models
    - fine-tuning, contains the jsonl files, created using the 1k samples for fine-tuning LLMs
    - with-emb, contains the datasets (i.e., drugbank and external datasets) with embeddings