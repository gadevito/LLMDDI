## SUPPLEMENTARY MATERIAL ##
This folder contains supplementary material. In particular, it contains:
- S.R3.2.1 Drug Pairs distribution analysis,  analyzes structural and biological features on 43,894 unique drug–drug pairs (21,947 positive interactions and 21,947 negatives) compiled from 13 external datasets in order to validate the negative sample strategy and check how different negatives and positives are.
- S.R6.2.1 Few-shot perturbation assesses the sensitivity of LLMs in few-shot settings to prompt and examples perturbations.
- S.R6.2.2 Few-shot ablation studies reports the ablation studies for LLMs in few-shot settings, considering SMILES only, genes only, species only, or pairwise combinations in prompts.
- S.R6.3.1 Fine-tuning computational info reports the computational details of the fine-tuning process.
- S.R6.3.2 Fine-tuning Phi3.5 on 2000 samples empirical validatates the choice of 1,000 samples for fine-tuning LLMs.
- S.R6.3.3 Fine-tuned LLM hit perturbations - Validation set assesses the robustness of fine-tuned LLMs against hit perturbations on the validation set.
- S.R6.4.1 Error Analysis GPT-4o vs PHI3.5 2.7B reports a qualitative error analysis across external datasets in the top-performing fine-tuned models, Phi-3.5 2.7B and GPT-4o. 
- S.R6.4.2 Fine-tuned LLMs Ablation studies conducts an ablation study for the fine-tuned version of Phi-3.5 2.7B using SMILES-only inputs, and on the fine-tuned version of GPT-4o using organisms-only inputs.
- S.R6.4.3 Fine-tuned LLMs hit perturbations - External Datasets assesses the robustness of fine-tuned LLMs against hit perturbations on the external datasets.
- S.R6.4.4 Fine-tuned LLMs Overlap analysis provides the results of the drug-entity overlap analysis between training and evaluation sets. 
- S.R6.4.5 Fine-tuned LLMs Prospective Analysis validates the negatives sample strategy analyzing prospectively, verifying whether any negative sample had been subsequently confirmed as interacting using an updated version of DrugBank (version 5.1.14 February 2025), released one year after our initial data collection.
- S.R6.4.6 Fine-tuned LLMs Label Noise reports a sensitivity analysis of the fine-tuned models by introducing 1%, 5%, and 10% random label flips for negative samples in the external datasets.
- S.R6.4.7 Fine-tuned LLMs Imbalanced Datasets results provides the detailed results for the evaluation of the fine-tuned LLMs' performance on naturally imbalanced dataset.
- S.R6.4.8 Fine-tuned GPT-4o Confidence Analysis analyzes the prediction confidence of the fine-tuned version of GPT-4o across all the validation and external datasets, in terms of log probabilities of the responses' tokens.
