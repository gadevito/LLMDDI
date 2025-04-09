# Exlore the pickle dataset created from the drugbank xml file and print statistics
import pickle
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import argparse



#
# First drugs data exploration.
# Checks: 'average_mass', 'monoisotopic_mass', 'calc_prop_molecular_formula', 'calc_prop_logp', 'exp_prop_logp'
#        'calc_prop_smiles', 'calc_prop_inchi' and FASTA sequences
#


def print_drug_info(drugs_df, drugbank_id):
    drug = drugs_df[drugs_df['drugbank_id'] == drugbank_id]
    if drug.empty:
        print(f"No drug found with drugbank_id: {drugbank_id}")
    else:
        print(f"Details for drugbank_id {drugbank_id}:")
        print(drug)

def print_drug_with_smiles(drugs_df, drugbank_id):
    drug = drugs_df[(drugs_df['drugbank_id'] == drugbank_id) & (drugs_df['calc_prop_smiles'].notnull())]
    if drug.empty:
        print(f"No drug with calc_prop_smiles found for drugbank_id: {drugbank_id}")
    else:
        print(f"Details for drugbank_id {drugbank_id} with non-null calc_prop_smiles:")
        print(drug, drug['calc_prop_smiles'])

def print_drugs_with_smiles(drugs_df):
    drug = drugs_df[(drugs_df['calc_prop_smiles'].notnull())]
    if drug.empty:
        print(f"No drug with calc_prop_smiles found")
    else:
        print(f"Details for drugbank_id with non-null calc_prop_smiles:")
        print(drug['drugbank_id'])


# Check if a specific drug has been withdrawn or is illicit, experimental, or investigational
def exclude_groups(groups):
    return not any(group in [3, 0, 5] for group in groups)

# Check if a drug belongs to pharmacological categories
def exclude_categories(categories):
    if categories is None or not isinstance(categories, list):  # Ensure it's a list
        return False
    return not any(category['category'] == "Non-Standardized Food Allergenic Extract" or category['category'] == "Non-Standardized Plant Allergenic Extract" for category in categories)

# Check if a drug has a FASTA sequence
def has_fasta_sequence(sequences):
    if not isinstance(sequences, list):  # Ensure it's a list
        return False
    return any(sequence.get('format') == 'FASTA' for sequence in sequences)

# Count FASTA sequences for a drug 
def count_fasta_sequences(sequences):
    if not isinstance(sequences, list):  # Ensure it's a list
        return 0
    return sum(1 for sequence in sequences if sequence.get('format') == 'FASTA')

def main(pickle_file,dont_save_ms):
    # Load the pickle file
    with open(pickle_file, 'rb') as f:
        drugs_data = pickle.load(f)

    # Convert drugs_data into a DataFrame
    drugs_df = pd.DataFrame(drugs_data)

    # Filter only valid drugs and categories
    filtered_drugs_df = drugs_df[
        drugs_df['groups'].apply(exclude_groups) &
        drugs_df['categories'].apply(exclude_categories)
    ]

    # Print the number of filtered drugs
    print(f"Number of drugs after filtering: {len(filtered_drugs_df)}")
    print(filtered_drugs_df.head())

    # Now, we work only with valid drugs
    drugs_df = filtered_drugs_df

    # First exploration
    print("First rows of the dataset:")
    print(drugs_df.head())

    print("\nColumns info:")
    print(drugs_df.info())

    # Print missing values per column 
    print("\nMissing values per column:")
    print(drugs_df.isnull().sum())

    # Analysis of Chemical Properties

    ## Molecular Mass
    print("\nDescriptive statistics for 'average_mass' and 'monoisotopic_mass':")
    print(drugs_df[['average_mass', 'monoisotopic_mass']].describe())

    # Visualize the distribution of average molecular mass
    plt.figure(figsize=(10, 6))
    sns.histplot(drugs_df['average_mass'].dropna(), bins=50, kde=True)
    plt.title('Distribution of Average Molecular Mass')
    plt.xlabel('Average Molecular Mass')
    plt.ylabel('Frequency')
    plt.show()

    ## Molecular formula
    print("\nDistribution of calculated molecular formulas:")
    print(drugs_df['calc_prop_molecular_formula'].value_counts().head())

    ## LogP
    print("\nDescriptive statistics for 'calc_prop_logp' and 'exp_prop_logp':")
    print(drugs_df[['calc_prop_logp', 'exp_prop_logp']].describe())

    # Visualize the relationship between calculated LogP and Average Molecular Mass
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x='calc_prop_logp', y='average_mass', data=drugs_df)
    plt.title('Relationship between Calculated LogP and Average Molecular Mass')
    plt.xlabel('Calculated LogP')
    plt.ylabel('Average Molecular Mass')
    plt.show()

    # Correlation analysis 
    correlation_matrix = drugs_df[['average_mass', 'monoisotopic_mass', 'calc_prop_logp', 'exp_prop_logp']].corr()
    plt.figure(figsize=(8, 6))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
    plt.title('Correlation Matrix of Chemical Properties')
    plt.show()

    total_drugs = len(drugs_df)

    # Molecular analysis
    num_missing_mol_formula = drugs_df['calc_prop_molecular_formula'].isnull().sum()
    print("\n=====================")
    print(f"Number of drugs without calculated molecular formula: {num_missing_mol_formula}")
    print(f"Percentage of drugs without calculated molecular formula: {num_missing_mol_formula / total_drugs * 100:.2f}%")


    # SMILE analysis
    print(drugs_df['calc_prop_smiles'].head())
    # Count the missing values in the column calc_prop_smiles.
    num_missing_smiles = drugs_df['calc_prop_smiles'].isnull().sum()
    

    print("\n=====================")
    print(f"Number of drugs without calculated SMILES: {num_missing_smiles}")
    print(f"Percentage of drugs without calculated SMILES: {num_missing_smiles / total_drugs * 100:.2f}%")

    # Filter drugs without SMILES or InChi or Fasta sequence or calc_prop_molecular_formula
    drugs_without_smiles_or_inchi = drugs_df[drugs_df['calc_prop_smiles'].isnull() | drugs_df['calc_prop_inchi'].isnull() | drugs_df['fasta_sequence'].isnull() | drugs_df['calc_prop_molecular_formula'].isnull()].copy()
    # Select the columns to export.
    subset_df = drugs_without_smiles_or_inchi[['drugbank_id', 'uniprot_id', 'kegg_drug_id', 'chebi_id', 'chembl_id', 'pubchem_compid', 'pubchem_subid', 'dpd_id', 'kegg_comp_id', 'chemspider_id', 'bindingdb_id', 'ndcd_id', 'genbank_id', 'ttd_id', 'pharmgkb_id', 'pdb_id', 'iuphar_id', 'gtp_id', 'zinc_id', 'rxcui_id', 'cas_number', 'unii', 'name', 'calc_prop_smiles', 'calc_prop_inchi', 'calc_prop_molecular_formula', 'calc_prop_molecular_weight','fasta_sequence' ]]    

    # Save the new dataset 
    if not dont_save_ms:
        subset_df.to_csv('drugs_without_mol_data.csv', index=False)


    # Filter drugs with SMILES but without calc_prop_logp
    drugs_with_smiles_no_logp = drugs_df[drugs_df['calc_prop_smiles'].notnull() & drugs_df['calc_prop_logp'].isnull()].copy()

    # Count and print the total count
    num_drugs_with_smiles_no_logp = len(drugs_with_smiles_no_logp)
    print("\n=====================")
    print(f"Number of drugs with SMILES but without calc_prop_logp: {num_drugs_with_smiles_no_logp}")

    # Calculate the number of drugs that do not have sequences in 'FASTA' format.
    #drugs_without_fasta = drugs_df[drugs_df['fasta_sequence'].isnull()].copy()
    no_fasta = drugs_df['fasta_sequence'].isnull().sum()

    # View the results
    print("\n=====================")
    print("Number of drugs without sequences in 'FASTA' format:", no_fasta)


    # InChI analysis
    print("\n=====================")
    num_missing_inchi = drugs_df['calc_prop_inchi'].isnull().sum()
    print(f"Number of drugs without calculated InChI: {num_missing_inchi}")
    print(f"Percentage of drugs without calculated InChI: {num_missing_inchi / total_drugs * 100:.2f}%")

    # Inchikey analysis
    print("\n=====================")
    num_missing_inchiKey = drugs_df['calc_prop_inchikey'].isnull().sum()
    print(f"Number of drugs without calculated InChIKey: {num_missing_inchiKey}")
    print(f"Percentage of drugs without calculated InChIKey: {num_missing_inchiKey / total_drugs * 100:.2f}%")


    # Molecular formula analysis
    no_mol_formula = drugs_df['calc_prop_molecular_formula'].isnull().sum()
    print(f"Number of drugs without calc_prop_molecular_formula: {no_mol_formula}")
    print(f"Percentage of drugs without calc_prop_molecular_formula: {no_mol_formula / total_drugs * 100:.2f}%")

    # IUPAC analysis 
    print("\n=====================")
    num_missing_iupac = drugs_df['calc_prop_iupac_name'].isnull().sum()
    print(f"Number of drugs without calculated iupac: {num_missing_iupac}")
    print(f"Percentage of drugs without calculated iupac: {num_missing_iupac / total_drugs * 100:.2f}%")


    # Filter drugs without SMILES, InChI, and FASTA sequences
    print("\n=====================")
    drugs_without_all = drugs_df[(drugs_df['calc_prop_smiles'].isnull()) & 
                                 (drugs_df['calc_prop_inchi'].isnull()) & 
                                 (drugs_df['fasta_sequence'].isnull())]

    print(f"Number of drugs without SMILES, InChI, and FASTA sequences: {len(drugs_without_all)}")


    # Filter drugs without calc_prop_molecular_formula, SMILES, InChI, and FASTA sequences
    print("\n=====================")
    drugs_without_all = drugs_df[(drugs_df['calc_prop_molecular_formula'].isnull()) &
                                 (drugs_df['calc_prop_smiles'].isnull()) & 
                                 (drugs_df['calc_prop_inchi'].isnull()) & 
                                 (drugs_df['fasta_sequence'].isnull())]

    print(f"Number of drugs without calc_prop_molecular_formula, SMILES, InChI, and FASTA sequences: {len(drugs_without_all)}")
    print(drugs_without_all.head())

    # Stampa delle sequenze dei primi 5 farmaci
    for index, row in drugs_without_all.head(5).iterrows():
        print(f"Drug {index + 1}:", row['drugbank_id'])
        sequences = row['sequences']
        if isinstance(sequences, list):
            for seq_info in sequences:
                print(f"Format: {seq_info.get('format')}, Sequence: {seq_info.get('sequence')}")
        else:
            print("No sequences available.")
        print("\n")

    print_drug_with_smiles(drugs_df, 'DB00014')
    print_drugs_with_smiles(drugs_df)
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='First Data Exploration')
    parser.add_argument('--pickle_file', type=str, required=True, help='Pickle file name containing the drug data.')
    parser.add_argument('--dont_save_ms', action='store_true',
                        help='Enable saving of missing smiles')

    args = parser.parse_args()
    main(args.pickle_file, args.dont_save_ms or False)