SYSTEM_DDI_CLASSIFICATION_RC ="You are an expert of drug-drug interaction. Given two drugs, where the order of administration counts, the genes and organisms targeted by the two drugs and the SMILES formulas of the two drugs, classify whether their administration causes 'interaction' or 'no interaction'. Answer only with the classification."


USER_DDI_CLASSIFICATION_SMILES ="""Drug1: {drug1}
SMILES for drug1: {smiles1}

Drug2: {drug2}
SMILES for drug2: {smiles2}

CLASSIFICATION:

*Answer only with the classification, no explanation or anything else*"""

USER_DDI_CLASSIFICATION_GENES ="""Drug1: {drug1}
Genes targeted by drug1: {genes1}

Drug2: {drug2}
Genes targeted by drug2: {genes2}

CLASSIFICATION:

*Answer only with the classification, no explanation or anything else*"""

USER_DDI_CLASSIFICATION_ORGS ="""Drug1: {drug1}
Organism targeted by drug1: {org1}

Drug2: {drug2}
Organism targeted by drug2: {org2}

CLASSIFICATION:

*Answer only with the classification, no explanation or anything else*"""

USER_DDI_CLASSIFICATION_SMILES_GENES ="""Drug1: {drug1}
SMILES for drug1: {smiles1}
Genes targeted by drug1: {genes1}

Drug2: {drug2}
SMILES for drug2: {smiles2}
Genes targeted by drug2: {genes2}

CLASSIFICATION:

*Answer only with the classification, no explanation or anything else*"""

USER_DDI_CLASSIFICATION_SMILES_ORGS ="""Drug1: {drug1}
SMILES for drug1: {smiles1}
Organism targeted by drug1: {org1}

Drug2: {drug2}
SMILES for drug2: {smiles2}
Organism targeted by drug2: {org2}

CLASSIFICATION:

*Answer only with the classification, no explanation or anything else*"""

USER_DDI_CLASSIFICATION_GENES_ORGS ="""Drug1: {drug1}
Organism targeted by drug1: {org1}
Genes targeted by drug1: {genes1}

Drug2: {drug2}
Organism targeted by drug2: {org2}
Genes targeted by drug2: {genes2}

CLASSIFICATION:

*Answer only with the classification, no explanation or anything else*"""