
SYSTEM_DDI_CLASSIFICATION_RC_SEMANTIC = """You are a drug-drug interaction specialist. Two drugs are given along with their administration order, molecular structures (SMILES), target organisms, and target genes. Determine if their combined administration results in 'interaction' or 'no interaction'. Provide only the classification."""


USER_DDI_CLASSIFICATION_RC_ORDER ="""## INSTRUCTIONS
- analyze the examples
- analyze the drug information
- answer only with the classification, no explanation or anything else.

Consider the following examples for correctly provide your answer:
{examples}

## CURRENT TASK INPUT DATA ##
Drug1: {drug1}
Genes targeted by drug1: {genes1}
Organism targeted by drug1: {org1}
SMILES for drug1: {smiles1}

Drug2: {drug2}
Genes targeted by drug2: {genes2}
Organism targeted by drug2: {org2}
SMILES for drug2: {smiles2}

CLASSIFICATION:

*Answer only with the classification, no explanation or anything else*"""