SYSTEM_DDI_CLASSIFICATION_RC =""


USER_DDI_CLASSIFICATION_RC ="""You are an AI assistant specialized in processing drug information based only on the provided details. Your task is to classify if a potential interaction might occur between two drugs based solely on comparing their SMILES, target genes, and organisms.

**CRITICAL OUTPUT REQUIREMENT:** Your *entire* response MUST be ONLY the single JSON object specified below. Absolutely NO extra text, NO explanations, NO notes, NO apologies, NO formatting (like ```json), NO repetition. Just the single JSON.

## OUTPUT FORMAT ##
{{"an":"Brief motivation based *strictly* on comparing SMILES, organisms, and genes targeted.", "c":"interaction" | "no interaction"}}

## CURRENT TASK INPUT DATA ##
Drug1 (Administered First): {drug1}
SMILES for drug1: {smiles1}
Organism targeted by drug1: {org1}
Genes targeted by drug1: {genes1}

Drug2 (Administered Second): {drug2}
SMILES for drug2: {smiles2}
Organism targeted by drug2: {org2}
Genes targeted by drug2: {genes2}

## GENERATE CLASSIFICATION NOW ##
Based *only* on the CURRENT TASK INPUT DATA, generate the SINGLE JSON classification. Remember: Adhere strictly to the OUTPUT FORMAT and the CRITICAL OUTPUT REQUIREMENT (JSON only, no extra text, no repetition). 
*Important:* DO NOT ADD ANYTHING ELSE AFTER THE JSON, no extra text, no explanation"""