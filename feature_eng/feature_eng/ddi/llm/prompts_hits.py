SYSTEM_DDI_CLASSIFICATION ="""You are an expert of drug-drug interaction. Given two drugs, where the order of administration counts, the genes and organisms targeted by the two drugs and the SMILES formulas of the two drugs, classify whether their administration causes 'interaction' or 'no interaction'. Answer only with the classification."""

SYSTEM_DDI_CLASSIFICATION_DS ="""You are an expert of drug-drug interaction. Given two drugs, where the order of administration counts, the genes and organisms targeted by the two drugs and the SMILES formulas of the two drugs, classify whether their administration causes 'interaction' or 'no interaction'. Answer only with the classification ('interaction' or 'no interaction'), nothing else."""


####### PERTURBATION #1 Changing order
USER_DDI_CLASSIFICATION_CHANGE_ORDER ="""Drug1: {drug1}
Genes targeted by drug1: {genes1}
Organism targeted by drug1: {org1}
SMILES for drug1: {smiles1}

Drug2: {drug2}
Genes targeted by drug2: {genes2}
Organism targeted by drug2: {org2}
SMILES for drug2: {smiles2}

CLASSIFICATION:"""

####### PERTURBATION #2 Sematic similar instructions

SYSTEM_DDI_CLASSIFICATION_SEMANTIC = """You are a drug-drug interaction specialist. Two drugs are given along with their administration order, molecular structures (SMILES), target organisms, and target genes. Determine if their combined administration results in 'interaction' or 'no interaction'. Provide only the classification."""

USER_DDI_CLASSIFICATION_SEMANTIC = """First drug: {drug1}
Molecular structure: {smiles1}
Target organism: {org1}
Target genes: {genes1}

Second drug: {drug2}
Molecular structure: {smiles2}
Target organism: {org2}
Target genes: {genes2}

PREDICTION:"""

####### PERTURBATION #3 List format

USER_DDI_CLASSIFICATION_LIST = """Drug Pair Information:

Drug A: {drug1}
- SMILES: {smiles1}
- Target Organism: {org1}
- Target Genes: {genes1}

Drug B: {drug2}
- SMILES: {smiles2}
- Target Organism: {org2}
- Target Genes: {genes2}

ANSWER:"""

####### PERTURBATION #4 Compact info
SYSTEM_DDI_CLASSIFICATION_COMPACT = """You are an expert of drug-drug interaction. Classify drug pairs as 'interaction' or 'no interaction' based on their properties. Answer only with the classification."""

USER_DDI_CLASSIFICATION_COMPACT = """DRUG 1: {drug1} | SMILES: {smiles1} | Organism: {org1} | Genes: {genes1}
DRUG 2: {drug2} | SMILES: {smiles2} | Organism: {org2} | Genes: {genes2}

CLASSIFICATION:"""


