## FEATURE ENGINEERING ##
This project contains the script used for feature engineering:
- first_data_exploration.py, explores data in the drugbank dataset.
- get_missed_formulas.py, uses external databases to gather the drugbank dataset missing information
- ddi:
    - gs_ddi.py, creates the compliant dataset, starting from the drugbank dataset, containing positive and negatives ddi instances
    - gs_ddi_split.py, splits the provided dataset in train and validation sets
    - ext_data_set_norm.py, reads the external datasets (the binary pickle versions), removes drugs which are not in drugbank and create the full ddi dataset, comprising all the external datasets ddi
    - gs_ddi_negs.py, creates a dataset containing only negative samples starting from the drugbank dataset.
    - gs_ddi_ext_datasets_db_negs.py, starts from the external datasets, and creates the dataset structure (compliant to our experiments) containing positives and negatives. The negative instances are extracted from an external pickle file, created from the drugbank dataset using gs_ddi_negs.py.
    - gs_extract_train_val.py, extracts training and validation samples from a dataset
    - gs_ddi_embeddings.py, updates the provided dataset computing embeddings for organism targeted by drugs and for SMILES formulas. We used MolFormer-XL for SMILES and text-embedding-3-small for organisms.
- llm:
    - prep_gs_ddi_GPT_ft.py, prepares the jsonl files for GPT-4 fine-tuning.
    - prep_gs_ddi_Gemma_ft.py, prepares the jsonl files for Gemma2 fine-tuning.
    - prep_gs_ddi_DS_ft.py, prepares the jsonl files for Deepseek r1 distilled models fine-tuning.
    - fine_tune_GPT.py, lauches the fine-tuning for GPT-4
    - gs_ddi_GPT_eval.py, evaluates the performance of the GPT-4 model on a secific dataset.
    - gs_ddi_Gemini_eval.py, evaluates the performance of the Gemini model on a secific dataset.
    - gs_ddi_Open_LLM_eval.py, evaluates the performance of Open LLM models on a secific dataset.
    - gs_ddi_Claude_eval.py, evaluates the performance of the Claude model on a secific dataset.
    - analyze_gs_ddi_LLM.py, compute metrics on the evaluation performed using eval scripts.

## Analysis scripts ##
In the main folder, feature_eng, you will find:
- distribution_analysis.py, analyzes the distribution of drug characterists reported in Section 3.2
- errors_for_groups.py, extracts data for the entity overlap analysis reported in RQ5
- token_distr_training_fs.py, assesses the token distributions for the similarity-based selection in few-shot setting

In the ddi/llm folder, you will find:
    - analyze_agreement.py, calculates the agreement percentage between two or more classifications
    - analyze_confidence.py, analyzes the confidence values (log probabilities) reported by an LLM (see RQ7).
    - analyze_gs_ddi_LLM.py, analyzes metrics for the classifications produced by LLMs
    - error_analysis.py, analyzes error patterns in fine-tuned LLMs (GPT-4o and PHI 3.5 2.7B), as reported in RQ7.

## Additional analyses ##
Ablation studies and perturbations are reproducible using the following scripts:
For fine-tuned LLMS, in the ddi/llm folder:
- flip_positives.py, applies random flips to negatives in a specific dataset
- gs_ddi_GPT_confidence_eval.py, performs GPT-4o inference and calculate log probabilities
- gs_ddi_GPT_LLM_eval_ft_ablation.py, performs inference using GPT-4o fine-tuned on organisms-only
- gs_ddi_Open_LLM_eval_ft_ablation.py, performs inference using PHI3.5 fined-tuned on SMILES
- gs_ddi_Open_LLM_eval_hits.py, perform inference on open-weight fine-tuned models using prompt hits
- gs_ddi_GPT_eval_hits.py, perform inference on GPT-4o fine-tuned using prompt hits

For few-shot settings, in the ddi/llm/fs folder:
- gs_ddi_Open_LLM_eval_ablation.py, performs inference using PHI3.5 in few-shot settings using single or pairwise drug characteristics
- gs_ddi_GPT_eval_ablation.py, performs inference using GPT-4o in few-shot settings using single or pairwise drug characteristics
- gs_ddi_Claude_eval_ablation.py, performs inference using Claude Sonnet 3.5 in few-shot settings using single or pairwise drug characteristics
- gs_ddi_Claude_eval_hits.py, performs inference using Claude Sonnet 3.5 in few-shot settings using prompt perturbations
- gs_ddi_GPT_eval_hits.py, performs inference using GPT-4o in few-shot settings using prompt perturbations