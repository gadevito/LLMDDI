SYSTEM_DDI_CLASSIFICATION ="""You are an expert of drug-drug interaction. Given two drugs, where the order of administration counts, the genes and organisms targeted by the two drugs and the SMILES formulas of the two drugs, classify whether their administration causes 'interaction' or 'no interaction'. Answer only with the classification."""

SYSTEM_DDI_CLASSIFICATION_DS ="""You are an expert of drug-drug interaction. Given two drugs, where the order of administration counts, the genes and organisms targeted by the two drugs and the SMILES formulas of the two drugs, classify whether their administration causes 'interaction' or 'no interaction'. Answer only with the classification ('interaction' or 'no interaction'), nothing else."""

USER_DDI_CLASSIFICATION ="""Drug1: {drug1}
SMILES for drug1: {smiles1}
Organism targeted by drug1: {org1}
Genes targeted by drug1: {genes1}

Drug2: {drug2}
SMILES for drug2: {smiles2}
Organism targeted by drug2: {org2}
Genes targeted by drug2: {genes2}

CLASSIFICATION:"""


