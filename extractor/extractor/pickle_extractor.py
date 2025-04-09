# Open the drugbank xml file and create a pickle file
import xml.etree.ElementTree as ET
import pickle
import argparse
from rdkit import Chem
from rdkit.Chem import Descriptors

# Calculate logp for smiles when missing
def calculate_logp(smiles):
    """
    Calculate logP using RDKit for a given SMILES
    """
    try:
        # Create an RDKit molecule for the given SMILES
        mol = Chem.MolFromSmiles(smiles)
        if mol is not None:
            # Calculate logP
            logp = Descriptors.MolLogP(mol)
            return logp
        return None
    except:
        return None
    
# Convert a string value to float
def to_float(val):
    return float(val) if val else None

# Convert (if possible) a string value to float
def to_float_safe(value):
    value_num = value
    if isinstance(value, str):
        value_clean = value.split()[0]
        try:
            value_num = float(value_clean)
        except ValueError:
            value_num = value

    return value_num

# One Hot encoding of group values
def encode_group(group):
    g = 7
    match group:
        case "approved":
            g = 1
        case "illicit":
            g = 0
        case "experimental":
            g = 2
        case "withdrawn":
            g = 3
        case "nutraceutical":
            g = 4
        case "investigational":
            g = 5
        case "vet_approved":
            g = 6
        case _:
            g = 7
    return g

# Remove spaces and "()" from a property name
def to_prop_name(kind):
    return kind.lower().replace(' ', '_').replace('(','').replace(')','')

# Experimental property names
experimental_props_names = ['exp_prop_boiling_point', 'exp_prop_water_solubility', 'exp_prop_pka', 'exp_prop_logp', 'exp_prop_molecular_formula', 'exp_prop_caco2_permeability', 'exp_prop_melting_point', 'exp_prop_molecular_weight', 'exp_prop_radioactivity', 'exp_prop_logs', 'exp_prop_isoelectric_point', 'exp_prop_hydrophobicity',
                            'exp_prop_boiling_point_source', 'exp_prop_water_solubility_source', 'exp_prop_pka_source', 'exp_prop_logp_source', 'exp_prop_molecular_formula_source', 'exp_prop_caco2_permeability_source', 'exp_prop_melting_point_source', 'exp_prop_molecular_weight_source', 'exp_prop_radioactivity_source', 'exp_prop_logs_source', 'exp_prop_isoelectric_point_source', 'exp_prop_hydrophobicity_source']

# Calculated property names
calculated_props_names = ['calc_prop_h_bond_acceptor_count', 'calc_prop_polar_surface_area_psa', 'calc_prop_h_bond_donor_count', 'calc_prop_mddr-like_rule', 'calc_prop_rule_of_five', 'calc_prop_rotatable_bond_count', 'calc_prop_number_of_rings', 'calc_prop_monoisotopic_weight', 'calc_prop_ghose_filter', 'calc_prop_bioavailability', 'calc_prop_smiles', 'calc_prop_logp', 'calc_prop_pka_strongest_acidic', 'calc_prop_inchikey', 'calc_prop_iupac_name', 'calc_prop_polarizability', 'calc_prop_logs', 'calc_prop_physiological_charge', 'calc_prop_molecular_weight', 'calc_prop_inchi', 'calc_prop_refractivity', 'calc_prop_water_solubility', 'calc_prop_traditional_iupac_name', 'calc_prop_pka_strongest_basic', 'calc_prop_molecular_formula',
                          'calc_prop_h_bond_acceptor_count_source', 'calc_prop_polar_surface_area_psa_source', 'calc_prop_h_bond_donor_count_source', 'calc_prop_mddr-like_rule_source', 'calc_prop_rule_of_five_source', 'calc_prop_rotatable_bond_count_source', 'calc_prop_number_of_rings_source', 'calc_prop_monoisotopic_weight_source', 'calc_prop_ghose_filter_source', 'calc_prop_bioavailability_source', 'calc_prop_smile_source', 'calc_prop_logp_source', 'calc_prop_pka_strongest_acidic_source', 'calc_prop_inchikey_source', 'calc_prop_iupac_name_source', 'calc_prop_polarizability_source', 'calc_prop_logs_source', 'calc_prop_physiological_charge_source', 'calc_prop_molecular_weight_source', 'calc_prop_inchi_source', 'calc_prop_refractivity_source', 'calc_prop_water_solubility_source', 'calc_prop_traditional_iupac_name_source', 'calc_prop_pka_strongest_basic_source', 'calc_prop_molecular_formula_source']

# Create a dictionary with all the property names
def prop_dictionary(names):
    return {key: None for key in names}

def align_properties(drug_data):
    if drug_data['calc_prop_logp'] is None:
        drug_data['calc_prop_logp'] = drug_data['exp_prop_logp']
    if drug_data['calc_prop_logs'] is None:
        drug_data['calc_prop_logs'] = drug_data['exp_prop_logs']    
    if drug_data['calc_prop_molecular_weight'] is None:
        drug_data['calc_prop_molecular_weight'] = drug_data['exp_prop_molecular_weight'] 
    if drug_data['calc_prop_water_solubility'] is None:
        drug_data['calc_prop_water_solubility'] = drug_data['exp_prop_water_solubility'] 
    if drug_data['calc_prop_molecular_formula'] is None:
        drug_data['calc_prop_molecular_formula'] = drug_data['exp_prop_molecular_formula']   

# Function to extract data from a "drug" xml element
def extract_drug_data(drug_element, max_samples=100000000):
    drug_data = {}

    
    #drug_data['drugbank_id'] = [id_elem.text for id_elem in drug_element.findall('{http://www.drugbank.ca}drugbank-id')]
    # primary="true" per selezionare solo l'attributo che rappresenta la chiave primaria

    # Drug ID and Base info
    drugbank_id_primary = None
    for id_elem in drug_element.findall('{http://www.drugbank.ca}drugbank-id'):
        if id_elem.attrib.get('primary') == 'true':
            drugbank_id_primary = id_elem.text
            break

    drug_data['drugbank_id'] = drugbank_id_primary
    drug_data['type'] = 1 if drug_element.get('type') =='small molecule' else 0 #biotech
    drug_data['created'] = drug_element.get('created')
    drug_data['updated'] = drug_element.get('updated')
    drug_data['name'] = drug_element.find('{http://www.drugbank.ca}name').text
    drug_data['description'] = drug_element.find('{http://www.drugbank.ca}description').text
    drug_data['cas_number'] = drug_element.find('{http://www.drugbank.ca}cas-number').text
    drug_data['unii'] = drug_element.find('{http://www.drugbank.ca}unii').text
    drug_data['average_mass'] = to_float(drug_element.find('{http://www.drugbank.ca}average-mass').text if drug_element.find('{http://www.drugbank.ca}average-mass') is not None else None)
    drug_data['monoisotopic_mass'] = to_float(drug_element.find('{http://www.drugbank.ca}monoisotopic-mass').text if drug_element.find('{http://www.drugbank.ca}monoisotopic-mass') is not None else None)
    state = drug_element.find('{http://www.drugbank.ca}state').text if drug_element.find('{http://www.drugbank.ca}state') is not None else None
    drug_data['state'] = 1 if state and state == 'solid' else 2 if state and state == 'liquid' else 3
    
    # Drug groups
    # Other groups that this drug belongs to. May include any of: approved, vet_approved, nutraceutical, illicit, withdrawn, investigational, and experimental.
    drug_data['groups'] = [encode_group(group_elem.text) for group_elem in drug_element.find('{http://www.drugbank.ca}groups').findall('{http://www.drugbank.ca}group')]

    # Affected organisms
    # Organisms in which the drug may display activity; activity may depend on local susceptibility patterns and resistance.
    affected_organisms = drug_element.find('{http://www.drugbank.ca}affected-organisms')
    if affected_organisms is not None:
        drug_data['affected_organisms'] = [organism.text for organism in affected_organisms.findall('{http://www.drugbank.ca}affected-organism')[:max_samples]]


    # General references to articles, links, and books.
    # In several locations within DrugBank XML files, a collection of references is provided. 
    # These references may be to articles, textbooks, or websites. Each of these categories of reference has a defined XML structure. 
    # A list of articles and textbooks used to inform the information provided about this drug.
    references = drug_element.find('{http://www.drugbank.ca}general-references')
    if references is not None:
        articles_elem = references.find('{http://www.drugbank.ca}articles')
        textbooks_elem = references.find('{http://www.drugbank.ca}textbooks')
        links_elem = references.find('{http://www.drugbank.ca}links')
        attachments_elem = references.find('{http://www.drugbank.ca}attachments')
        drug_data['general_references'] = {
            'articles': [
                (
                    article.find('pubmed-id').text if article.find('pubmed-id') is not None else None,
                    article.find('citation').text if article.find('citation') is not None else None
                ) for article in articles_elem.findall('{http://www.drugbank.ca}article')[:max_samples]
            ] if articles_elem is not None else [],
            'textbooks': [
                (
                    textbook.find('isbn').text if textbook.find('isbn') is not None else None,
                    textbook.find('citation').text if textbook.find('citation') is not None else None
                ) for textbook in textbooks_elem.findall('{http://www.drugbank.ca}textbook')[:max_samples] if textbooks_elem is not None
            ] if textbooks_elem is not None else [],
            'links': [
                (
                    link.find('title').text if link.find('title') is not None else None,
                    link.find('url').text if link.find('url') is not None else None
                ) for link in links_elem.findall('{http://www.drugbank.ca}link')[:max_samples] if links_elem is not None
            ] if links_elem is not None else [],
            'attachments': [
                (
                    attachment.find('title').text if attachment.find('title') is not None else None,
                    attachment.find('url').text if attachment.find('url') is not None else None
                ) for attachment in attachments_elem.findall('{http://www.drugbank.ca}attachment')[:max_samples] if attachments_elem is not None
            ] if attachments_elem is not None else []        
        }

    # Citation for synthesis of the drug molecule
    drug_data['synthesis_reference'] = drug_element.find('{http://www.drugbank.ca}synthesis-reference').text

    # Pharmacology
    # Describes the use, mechanism of action, pharmacokinetics, pharmacodynamics, and physiological or biochemical effects in the body.
    drug_data['indication'] = drug_element.find('{http://www.drugbank.ca}indication').text
    drug_data['pharmacodynamics'] = drug_element.find('{http://www.drugbank.ca}pharmacodynamics').text
    drug_data['mechanism_of_action'] = drug_element.find('{http://www.drugbank.ca}mechanism-of-action').text
    drug_data['toxicity'] = drug_element.find('{http://www.drugbank.ca}toxicity').text
    drug_data['metabolism'] = drug_element.find('{http://www.drugbank.ca}metabolism').text
    drug_data['absorption'] = drug_element.find('{http://www.drugbank.ca}absorption').text
    drug_data['half_life'] = drug_element.find('{http://www.drugbank.ca}half-life').text
    drug_data['protein_binding'] = drug_element.find('{http://www.drugbank.ca}protein-binding').text
    drug_data['route_of_elimination'] = drug_element.find('{http://www.drugbank.ca}route-of-elimination').text
    drug_data['volume_of_distribution'] = drug_element.find('{http://www.drugbank.ca}volume-of-distribution').text
    drug_data['clearance'] = drug_element.find('{http://www.drugbank.ca}clearance').text
    
    # Drug Classification
    # A description of the hierarchical chemical classification of the drug; imported from ClassyFire.
    classification = drug_element.find('{http://www.drugbank.ca}classification')
    if classification is not None:
        drug_data['classification'] = {
            'description': classification.find('{http://www.drugbank.ca}description').text,
            'direct_parent': classification.find('{http://www.drugbank.ca}direct-parent').text,
            'kingdom': classification.find('{http://www.drugbank.ca}kingdom').text,
            'superclass': classification.find('{http://www.drugbank.ca}superclass').text,
            'class': classification.find('{http://www.drugbank.ca}class').text,
            'subclass': classification.find('{http://www.drugbank.ca}subclass').text,
            'alternative_parents': [alt_parent.text for alt_parent in classification.findall('{http://www.drugbank.ca}alternative-parent')],
            'substituents': [substituent.text for substituent in classification.findall('{http://www.drugbank.ca}substituent')]
        }
    
    # Salts
    # Available salt forms of the drug. Ions such as hydrochloride, sodium, and sulfate are often added to the drug molecule to increase solubility, dissolution, or absorption.
    salts = drug_element.find('{http://www.drugbank.ca}salts')
    if salts is not None:
        drug_data['salts'] = [{
            'drugbank_id': next((drugbank_id.text 
                            for drugbank_id in salt.findall('{http://www.drugbank.ca}drugbank-id') 
                            if drugbank_id.get('primary', 'false') == 'true'), None),
            'name': salt.find('{http://www.drugbank.ca}name').text,
            'unii': salt.find('{http://www.drugbank.ca}unii').text,
            'cas_number': salt.find('{http://www.drugbank.ca}cas-number').text,
            'inchikey': salt.find('{http://www.drugbank.ca}inchikey').text,
            'average_mass': salt.find('{http://www.drugbank.ca}average-mass').text if salt.find('{http://www.drugbank.ca}average-mass') is not None else None,
            'monoisotopic_mass': salt.find('{http://www.drugbank.ca}monoisotopic-mass').text if salt.find('{http://www.drugbank.ca}monoisotopic-mass') is not None else None
        } for salt in salts.findall('{http://www.drugbank.ca}salt')[:max_samples]]
    
    # Synonyms
    # Other names or identifiers that are associated with this drug.
    synonyms = drug_element.find('{http://www.drugbank.ca}synonyms')
    if synonyms is not None:
        drug_data['synonyms'] = [{
            'synonym': synonym.text,
            'language': synonym.get('language'),
            'coder': synonym.get('coder')
        } for synonym in synonyms.findall('{http://www.drugbank.ca}synonym')[:max_samples]]
    
    # Associated Products
    # A list of commercially available products in Canada and the United States that contain the drug.
    products = drug_element.find('{http://www.drugbank.ca}products')
    if products is not None:
        drug_data['products'] = [{
            'name': product.find('{http://www.drugbank.ca}name').text,
            'labeller': product.find('{http://www.drugbank.ca}labeller').text,
            'ndc_id': product.find('{http://www.drugbank.ca}ndc-id').text,
            'ndc_product_code': product.find('{http://www.drugbank.ca}ndc-product-code').text,
            'dpd_id': product.find('{http://www.drugbank.ca}dpd-id').text,
            'ema_product_code': product.find('{http://www.drugbank.ca}ema-product-code').text,
            'ema_ma_number': product.find('{http://www.drugbank.ca}ema-ma-number').text,
            'started_marketing_on': product.find('{http://www.drugbank.ca}started-marketing-on').text,
            'ended_marketing_on': product.find('{http://www.drugbank.ca}ended-marketing-on').text,
            'dosage_form': product.find('{http://www.drugbank.ca}dosage-form').text,
            'strength': product.find('{http://www.drugbank.ca}strength').text,
            'route': product.find('{http://www.drugbank.ca}route').text,
            'fda_application_number': product.find('{http://www.drugbank.ca}fda-application-number').text,
            'generic': product.find('{http://www.drugbank.ca}generic').text,
            'over_the_counter': product.find('{http://www.drugbank.ca}over-the-counter').text,
            'approved': product.find('{http://www.drugbank.ca}approved').text,
            'country': product.find('{http://www.drugbank.ca}country').text,
            'source': product.find('{http://www.drugbank.ca}source').text
        } for product in products.findall('{http://www.drugbank.ca}product')[:max_samples]]
    
    # International Brands
    # The proprietary names used by the manufacturers for commercially available forms of the drug, focusing on brand names for products that are available in countries other than Canada and the Unites States.
    international_brands = drug_element.find('{http://www.drugbank.ca}international-brands')
    if international_brands is not None:
        drug_data['international_brands'] = [{
            'name': brand.find('{http://www.drugbank.ca}name').text,
            'company': brand.find('{http://www.drugbank.ca}company').text
        } for brand in international_brands.findall('{http://www.drugbank.ca}international-brand')[:max_samples]]
    
    # Mixtures
    # All commercially available products in which this drug is available in combination with other drug molecules.
    mixtures = drug_element.find('{http://www.drugbank.ca}mixtures')
    if mixtures is not None:
        drug_data['mixtures'] = [{
            'name': mixture.find('{http://www.drugbank.ca}name').text,
            'ingredients': mixture.find('{http://www.drugbank.ca}ingredients').text,
            'supplemental_ingredients': mixture.find('{http://www.drugbank.ca}supplemental-ingredients').text if mixture.find('{http://www.drugbank.ca}supplemental-ingredients') is not None else None
        } for mixture in mixtures.findall('{http://www.drugbank.ca}mixture')[:max_samples]]
    
    # Packagers
    # A list of companies that are packaging the drug for re-distribution.
    packagers = drug_element.find('{http://www.drugbank.ca}packagers')
    if packagers is not None:
        drug_data['packagers'] = [{
            'name': packager.find('{http://www.drugbank.ca}name').text,
            'url': packager.find('{http://www.drugbank.ca}url').text
        } for packager in packagers.findall('{http://www.drugbank.ca}packager')[:max_samples]]

    # Manufacturers
    # A list of companies that are manufacturing the commercially available forms of this drug that are available in Canada and the Unites States.
    manufacturers = drug_element.find('{http://www.drugbank.ca}manufacturers')
    if manufacturers is not None:
        drug_data['manufacturers'] = [{
            'name': manufacturer.text,
            'generic': manufacturer.get('generic'),
            'url': manufacturer.get('url')
        } for manufacturer in manufacturers.findall('{http://www.drugbank.ca}manufacturer')[:max_samples]]

    # Prices
    # Unit drug prices
    prices = drug_element.find('{http://www.drugbank.ca}prices')
    if prices is not None:
        drug_data['prices'] = [{
            'description': price.find('{http://www.drugbank.ca}description').text,
            'cost': {
                'value': price.find('{http://www.drugbank.ca}cost').text,
                'currency': price.find('{http://www.drugbank.ca}cost').get('currency')
            },
            'unit': price.find('{http://www.drugbank.ca}unit').text
        } for price in prices.findall('{http://www.drugbank.ca}price')[:max_samples]]

    # Categories
    # General categorizations of the drug.
    categories = drug_element.find('{http://www.drugbank.ca}categories')
    if categories is not None:
        drug_data['categories'] = [{
            'category': category.find('{http://www.drugbank.ca}category').text,
            'mesh_id': category.find('{http://www.drugbank.ca}mesh-id').text
        } for category in categories.findall('{http://www.drugbank.ca}category')[:max_samples]]


    # Dosages
    # A list of the commercially available dosages of the drug.
    dosages = drug_element.find('{http://www.drugbank.ca}dosages')
    if dosages is not None:
        drug_data['dosages'] = [{
            'form': dosage.find('{http://www.drugbank.ca}form').text,
            'route': dosage.find('{http://www.drugbank.ca}route').text,
            'strength': dosage.find('{http://www.drugbank.ca}strength').text
        } for dosage in dosages.findall('{http://www.drugbank.ca}dosage')[:max_samples]]

    # ATC codes
    # The Anatomical Therapeutic Classification (ATC) code for the drug assigned by the World Health Organization Anatomical Chemical Classification System.
    atc_codes = drug_element.find('{http://www.drugbank.ca}atc-codes')
    if atc_codes is not None:
        drug_data['atc_codes'] = [{
            'code': atc_code.get('code'),
            'levels': [{
                'level': level.text,
                'code': level.get('code')
            } for level in atc_code.findall('{http://www.drugbank.ca}level')]
        } for atc_code in atc_codes.findall('{http://www.drugbank.ca}atc-code')[:max_samples]]

    # AHFS codes
    # The American Hospital Formulary Service (AHFS) identifier for this drug.
    ahfs_codes = drug_element.find('{http://www.drugbank.ca}ahfs-codes')
    if ahfs_codes is not None:
        drug_data['ahfs_codes'] = [code.text for code in ahfs_codes.findall('{http://www.drugbank.ca}ahfs-code')[:max_samples]]

    # PDB Entries
    # Protein Data Bank (PDB) identifiers for this drug.
    pdb_entries = drug_element.find('{http://www.drugbank.ca}pdb-entries')
    if pdb_entries is not None:
        drug_data['pdb_entries'] = [entry.text for entry in pdb_entries.findall('{http://www.drugbank.ca}pdb-entry')[:max_samples]]

    # FDA label
    # Contains a URL for accessing the uploaded United States Food and Drug Administration (FDA) Monograph for this drug.
    drug_data['fda_label'] = drug_element.find('{http://www.drugbank.ca}fda-label').text if drug_element.find('{http://www.drugbank.ca}fda-label') is not None else None

    # MSDS
    # Contains a URL for accessing the Material Safety Data Sheet (MSDS) for this drug.
    drug_data['msds'] = drug_element.find('{http://www.drugbank.ca}msds').text if drug_element.find('{http://www.drugbank.ca}msds') is not None else None

    # Patents
    # A property right issued by the United States Patent and Trademark Office (USPTO) to an inventor for a limited time, in exchange for public disclosure of the invention when the patent is granted. 
    # Drugs may be issued multiple patents.
    patents = drug_element.find('{http://www.drugbank.ca}patents')
    if patents is not None:
        drug_data['patents'] = [{
            'number': patent.find('{http://www.drugbank.ca}number').text,
            'country': patent.find('{http://www.drugbank.ca}country').text,
            'approved': patent.find('{http://www.drugbank.ca}approved').text,
            'expires': patent.find('{http://www.drugbank.ca}expires').text,
            'pediatric_extension': patent.find('{http://www.drugbank.ca}pediatric-extension').text
        } for patent in patents.findall('{http://www.drugbank.ca}patent')[:max_samples]]

    # Food interactions
    # Food that may interact with this drug.
    food_interactions = drug_element.find('{http://www.drugbank.ca}food-interactions')
    if food_interactions is not None:
        drug_data['food_interactions'] = [food.text for food in food_interactions.findall('{http://www.drugbank.ca}food-interaction')[:max_samples]]

    # Drug interactions
    # Drug-drug interactions detailing drugs that, when administered concomitantly with the drug of interest, will affect its activity or result in adverse effects. 
    # These interactions may be synergistic or antagonistic depending on the physiological effects and mechanism of action of each drug.
    drug_interactions = drug_element.find('{http://www.drugbank.ca}drug-interactions')
    if drug_interactions is not None:
        drug_data['drug_interactions'] = [{
            'drugbank_id': interaction.find('{http://www.drugbank.ca}drugbank-id').text,
            'name': interaction.find('{http://www.drugbank.ca}name').text,
            'description': interaction.find('{http://www.drugbank.ca}description').text
        } for interaction in drug_interactions.findall('{http://www.drugbank.ca}drug-interaction')[:max_samples]]   
    
    # Sequences
    # The amino acid sequence; provided if the drug is a peptide.
    sequences = drug_element.find('{http://www.drugbank.ca}sequences')
    drug_data['fasta_sequence'] = None
    if sequences is not None:
        fasta_seq = []
        drug_data['sequences'] = []

        # For multiple FASTA format, we concatenate them
        for sequence in sequences.findall('{http://www.drugbank.ca}sequence')[:max_samples]:
            format = sequence.get('format')
            sequence = sequence.text
            if format == 'FASTA' or format is None:
                fasta_seq.append(sequence)
            else:
                drug_data['sequences'].append({'sequence':sequence, 'format': format})
        if fasta_seq and len(fasta_seq)>0:
            drug_data['fasta_sequence'] = "\n".join(fasta_seq)
            #print(drug_data['fasta_sequence'])
            drug_data['sequences'].append({'sequence':"\n".join(fasta_seq), 'format':'FASTA'})


        #drug_data['sequences'] = [{
        #    'sequence': sequence.text,
        #    'format': sequence.get('format')
        #} for sequence in sequences.findall('{http://www.drugbank.ca}sequence')[:max_samples]]

    # Calculated properties
    # Drug properties that have been predicted by ChemAxon or ALOGPS based on the inputed chemical structure.
    calculated_properties = drug_element.find('{http://www.drugbank.ca}calculated-properties')
    if calculated_properties is not None:
        calc_p = prop_dictionary(calculated_props_names)
        for prop in calculated_properties.findall('{http://www.drugbank.ca}property')[:max_samples]:

            kind = prop.find('{http://www.drugbank.ca}kind').text
            kind = to_prop_name(kind)
            name = f'calc_prop_{kind}'

            if name != 'calc_prop_smile':
                calc_p[name] = to_float_safe(prop.find('{http://www.drugbank.ca}value').text)
            else:
                calc_p[name] = prop.find('{http://www.drugbank.ca}value').text
            source = f'calc_prop_{kind}_source'
            v_source = prop.find('{http://www.drugbank.ca}source').text
            calc_p[source] = 1 if v_source and v_source == "ChemAxon" else 0

        if calc_p['calc_prop_smiles'] and calc_p['calc_prop_logp'] is None:
            print("Non c'è, lo calcolo", calc_p['calc_prop_smiles'], calc_p['calc_prop_logp'])
            calc_p['calc_prop_logp'] = calculate_logp(calc_p['calc_prop_smiles'])
            print(calc_p['calc_prop_logp'])
            calc_p['calc_prop_logp_source'] = 0

        drug_data.update(calc_p)
    else:
        calc_p = prop_dictionary(calculated_props_names)
        drug_data.update(calc_p)

        #drug_data['calculated_properties'] = [{
        #    'kind': prop.find('{http://www.drugbank.ca}kind').text,
        #    'value': prop.find('{http://www.drugbank.ca}value').text,
        #    'source': prop.find('{http://www.drugbank.ca}source').text
        #} for prop in calculated_properties.findall('{http://www.drugbank.ca}property')[:max_samples]]

    # Experimental properties
    # Drug properties that have been experimentally proven.
    experimental_properties = drug_element.find('{http://www.drugbank.ca}experimental-properties')
    if experimental_properties is not None:

        calc_p = prop_dictionary(experimental_props_names)
        for prop in experimental_properties.findall('{http://www.drugbank.ca}property')[:max_samples]:
            kind = prop.find('{http://www.drugbank.ca}kind').text
            kind = to_prop_name(kind)
            name = f'exp_prop_{kind}'
            calc_p[name] = to_float_safe(prop.find('{http://www.drugbank.ca}value').text)
            source = f'exp_prop_{kind}_source'
            calc_p[source] = prop.find('{http://www.drugbank.ca}source').text

        drug_data.update(calc_p)

        #drug_data['experimental_properties'] = [{
        #    'kind': prop.find('{http://www.drugbank.ca}kind').text,
        #    'value': prop.find('{http://www.drugbank.ca}value').text,
        #    'source': prop.find('{http://www.drugbank.ca}source').text
        #} for prop in experimental_properties.findall('{http://www.drugbank.ca}property')[:max_samples]]
    else:
        calc_p = prop_dictionary(experimental_props_names)
        drug_data.update(calc_p)

    align_properties(drug_data)

    # External identifiers
    # Identifiers used in other websites or databases providing information about this drug.
    external_identifiers = drug_element.find('{http://www.drugbank.ca}external-identifiers')
    drug_data["uniprot_id"] = None
    drug_data["kegg_drug_id"] = None
    drug_data["chebi_id"] = None
    drug_data["chembl_id"] = None
    drug_data["pubchem_compid"] = None
    drug_data["pubchem_subid"] = None
    drug_data["dpd_id"] = None
    drug_data["kegg_comp_id"] = None
    drug_data["chemspider_id"] = None
    drug_data["bindingdb_id"] = None
    drug_data["ndcd_id"] = None
    drug_data["genbank_id"] = None
    drug_data["ttd_id"] = None
    drug_data["pharmgkb_id"] = None
    drug_data["pdb_id"] = None
    drug_data["iuphar_id"] = None
    drug_data["gtp_id"] = None
    drug_data["zinc_id"] = None
    drug_data["rxcui_id"] = None

    if external_identifiers is not None:
        drug_data['external_identifiers'] = [{
            'resource': ext_id.find('{http://www.drugbank.ca}resource').text,
            'identifier': ext_id.find('{http://www.drugbank.ca}identifier').text
        } for ext_id in external_identifiers.findall('{http://www.drugbank.ca}external-identifier')[:max_samples]]
        #UniProtKB
        for ext_id in external_identifiers.findall('{http://www.drugbank.ca}external-identifier')[:max_samples]:
            if ext_id.find('{http://www.drugbank.ca}resource').text == "UniProtKB":
                drug_data["uniprot_id"] = ext_id.find('{http://www.drugbank.ca}identifier').text
            elif ext_id.find('{http://www.drugbank.ca}resource').text =="KEGG Drug":
                drug_data["kegg_drug_id"] = ext_id.find('{http://www.drugbank.ca}identifier').text
            elif ext_id.find('{http://www.drugbank.ca}resource').text =="ChEBI":
                drug_data["chebi_id"] = ext_id.find('{http://www.drugbank.ca}identifier').text
            elif ext_id.find('{http://www.drugbank.ca}resource').text =="ChEMBL":
                drug_data["chembl_id"] = ext_id.find('{http://www.drugbank.ca}identifier').text
            elif ext_id.find('{http://www.drugbank.ca}resource').text =="PubChem Compound":
                drug_data["pubchem_compid"] = ext_id.find('{http://www.drugbank.ca}identifier').text
            elif ext_id.find('{http://www.drugbank.ca}resource').text =="PubChem Substance":
                drug_data["pubchem_subid"] = ext_id.find('{http://www.drugbank.ca}identifier').text
            elif ext_id.find('{http://www.drugbank.ca}resource').text =="Drugs Product Database (DPD)":
                drug_data["dpd_id"] = ext_id.find('{http://www.drugbank.ca}identifier').text
            elif ext_id.find('{http://www.drugbank.ca}resource').text =="KEGG Compound":
                drug_data["kegg_comp_id"] = ext_id.find('{http://www.drugbank.ca}identifier').text
            elif ext_id.find('{http://www.drugbank.ca}resource').text =="ChemSpider":
                drug_data["chemspider_id"] = ext_id.find('{http://www.drugbank.ca}identifier').text
            elif ext_id.find('{http://www.drugbank.ca}resource').text =="BindingDB":
                drug_data["bindingdb_id"] = ext_id.find('{http://www.drugbank.ca}identifier').text
            elif ext_id.find('{http://www.drugbank.ca}resource').text =="National Drug Code Directory":
                drug_data["ndcd_id"] = ext_id.find('{http://www.drugbank.ca}identifier').text
            elif ext_id.find('{http://www.drugbank.ca}resource').text =="GenBank":
                drug_data["genbank_id"] = ext_id.find('{http://www.drugbank.ca}identifier').text
            elif ext_id.find('{http://www.drugbank.ca}resource').text =="Therapeutic Targets Database":
                drug_data["ttd_id"] = ext_id.find('{http://www.drugbank.ca}identifier').text
            elif ext_id.find('{http://www.drugbank.ca}resource').text =="PharmGKB":
                drug_data["pharmgkb_id"] = ext_id.find('{http://www.drugbank.ca}identifier').text
            elif ext_id.find('{http://www.drugbank.ca}resource').text =="PDB":
                drug_data["pdb_id"] = ext_id.find('{http://www.drugbank.ca}identifier').text
            elif ext_id.find('{http://www.drugbank.ca}resource').text =="IUPHAR":
                drug_data["iuphar_id"] = ext_id.find('{http://www.drugbank.ca}identifier').text
            elif ext_id.find('{http://www.drugbank.ca}resource').text =="Guide to Pharmacology":
                drug_data["gtp_id"] = ext_id.find('{http://www.drugbank.ca}identifier').text
            elif ext_id.find('{http://www.drugbank.ca}resource').text =="ZINC":
                drug_data["zinc_id"] = ext_id.find('{http://www.drugbank.ca}identifier').text
            elif ext_id.find('{http://www.drugbank.ca}resource').text =="RxCUI":
                drug_data["rxcui_id"] = ext_id.find('{http://www.drugbank.ca}identifier').text

    # External links
    # Links to other websites or databases providing information about this drug.
    external_links = drug_element.find('{http://www.drugbank.ca}external-links')
    if external_links is not None:
        drug_data['external_links'] = [{
            'resource': link.find('{http://www.drugbank.ca}resource').text,
            'url': link.find('{http://www.drugbank.ca}url').text
        } for link in external_links.findall('{http://www.drugbank.ca}external-link')[:max_samples]]

    # Pathways
    # Metabolic, disease, and biological pathways that the drug is involved in, as identified by the Small Molecule Pathway Database (SMPDB).
    pathways = drug_element.find('{http://www.drugbank.ca}pathways')
    if pathways is not None:
        drug_data['pathways'] = [{
            'smpdb_id': pathway.find('{http://www.drugbank.ca}smpdb-id').text,
            'name': pathway.find('{http://www.drugbank.ca}name').text,
            'category': pathway.find('{http://www.drugbank.ca}category').text,
            'drugs': [{
                'drugbank_id': drug.find('{http://www.drugbank.ca}drugbank-id').text,
                'name': drug.find('{http://www.drugbank.ca}name').text
            } for drug in pathway.find('{http://www.drugbank.ca}drugs').findall('{http://www.drugbank.ca}drug')],
            'enzymes': [enzyme.text for enzyme in pathway.find('{http://www.drugbank.ca}enzymes').findall('{http://www.drugbank.ca}uniprot-id')]
        } for pathway in pathways.findall('{http://www.drugbank.ca}pathway')[:max_samples]]  
    
    # Metabolic Reactions
    # A sequential representation of the metabolic reactions that this drug molecule is involved in. 
    # Depending on available information, this may include metabolizing enzymes, reaction type, substrates, products, pharmacological activity of metabolites, and a structural representation of the biochemical reactions.
    reactions = drug_element.find('{http://www.drugbank.ca}reactions')
    if reactions is not None:
        drug_data['reactions'] = [{
            'sequence': reaction.find('{http://www.drugbank.ca}sequence').text,
            'left_element': {
                'drugbank_id': reaction.find('{http://www.drugbank.ca}left-element').find('{http://www.drugbank.ca}drugbank-id').text,
                'name': reaction.find('{http://www.drugbank.ca}left-element').find('{http://www.drugbank.ca}name').text
            },
            'right_element': {
                'drugbank_id': reaction.find('{http://www.drugbank.ca}right-element').find('{http://www.drugbank.ca}drugbank-id').text,
                'name': reaction.find('{http://www.drugbank.ca}right-element').find('{http://www.drugbank.ca}name').text
            },
            'enzymes': [{
                'drugbank_id': enzyme.find('{http://www.drugbank.ca}drugbank-id').text,
                'name': enzyme.find('{http://www.drugbank.ca}name').text,
                'uniprot_id': enzyme.find('{http://www.drugbank.ca}uniprot-id').text
            } for enzyme in reaction.find('{http://www.drugbank.ca}enzymes').findall('{http://www.drugbank.ca}enzyme')]
        } for reaction in reactions.findall('{http://www.drugbank.ca}reaction')[:max_samples]]
    
    # SNP interactions
    # A list of single nucleotide polymorphisms (SNPs) relevent to drug activity or metabolism, 
    # and the effects these may have on pharmacological activity. 
    # SNP effects in the patient may require close monitoring, an increase or decrease in dose, or a change in therapy.
    snp_effects = drug_element.find('{http://www.drugbank.ca}snp-effects')
    if snp_effects is not None:
        drug_data['snp_effects'] = [{
            'protein_name': effect.find('{http://www.drugbank.ca}protein-name').text,
            'gene_symbol': effect.find('{http://www.drugbank.ca}gene-symbol').text,
            'uniprot_id': effect.find('{http://www.drugbank.ca}uniprot-id').text,
            'rs_id': effect.find('{http://www.drugbank.ca}rs-id').text,
            'allele': effect.find('{http://www.drugbank.ca}allele').text,
            'defining_change': effect.find('{http://www.drugbank.ca}defining-change').text,
            'description': effect.find('{http://www.drugbank.ca}description').text,
            'pubmed_id': effect.find('{http://www.drugbank.ca}pubmed-id').text
        } for effect in snp_effects.findall('{http://www.drugbank.ca}effect')[:max_samples]]

    # SNP adverse drug reactions
    # The adverse drug reactions that may occur as a result of the listed single nucleotide polymorphisms (SNPs).
    snp_adverse_drug_reactions = drug_element.find('{http://www.drugbank.ca}snp-adverse-drug-reactions')
    if snp_adverse_drug_reactions is not None:
        drug_data['snp_adverse_drug_reactions'] = [{
            'protein_name': reaction.find('{http://www.drugbank.ca}protein-name').text,
            'gene_symbol': reaction.find('{http://www.drugbank.ca}gene-symbol').text,
            'uniprot_id': reaction.find('{http://www.drugbank.ca}uniprot-id').text,
            'rs_id': reaction.find('{http://www.drugbank.ca}rs-id').text,
            'allele': reaction.find('{http://www.drugbank.ca}allele').text,
            'adverse_reaction': reaction.find('{http://www.drugbank.ca}adverse-reaction').text,
            'description': reaction.find('{http://www.drugbank.ca}description').text,
            'pubmed_id': reaction.find('{http://www.drugbank.ca}pubmed-id').text
        } for reaction in snp_adverse_drug_reactions.findall('{http://www.drugbank.ca}reaction')[:max_samples]]


    # Targets
    # Protein targets of drug action, enzymes that are inhibited/induced or involved in metabolism, and carrier or transporter proteins involved in movement of the drug across biological membranes.
    targets = drug_element.find('{http://www.drugbank.ca}targets')
    if targets is not None:
        drug_data['targets'] = [{
            'id': target.find('{http://www.drugbank.ca}id').text,
            'name': target.find('{http://www.drugbank.ca}name').text,
            'organism': target.find('{http://www.drugbank.ca}organism').text,
            'actions': [action.text for action in target.find('{http://www.drugbank.ca}actions').findall('{http://www.drugbank.ca}action')],
            'known_action': target.find('{http://www.drugbank.ca}known-action').text,
            'polypeptides': [{
                'id': polypeptide.attrib.get('id'),
                'name': polypeptide.find('{http://www.drugbank.ca}name').text,
                'organism': polypeptide.find('{http://www.drugbank.ca}organism').text,
                'general_function': polypeptide.find('{http://www.drugbank.ca}general-function').text,
                'specific_function': polypeptide.find('{http://www.drugbank.ca}specific-function').text,
                'gene_name': polypeptide.find('{http://www.drugbank.ca}gene-name').text,
                'locus': polypeptide.find('{http://www.drugbank.ca}locus').text,
                'cellular_location': polypeptide.find('{http://www.drugbank.ca}cellular-location').text,
                'transmembrane_regions': polypeptide.find('{http://www.drugbank.ca}transmembrane-regions').text,
                'signal_regions': polypeptide.find('{http://www.drugbank.ca}signal-regions').text,
                'theoretical_pi': polypeptide.find('{http://www.drugbank.ca}theoretical-pi').text,
                'molecular_weight': polypeptide.find('{http://www.drugbank.ca}molecular-weight').text,
                'chromosome_location': polypeptide.find('{http://www.drugbank.ca}chromosome-location').text,
                'external_identifiers': [{
                    'resource': ext_id.find('{http://www.drugbank.ca}resource').text,
                    'identifier': ext_id.find('{http://www.drugbank.ca}identifier').text
                } for ext_id in polypeptide.find('{http://www.drugbank.ca}external-identifiers').findall('{http://www.drugbank.ca}external-identifier')],
                'synonyms': [synonym.text for synonym in polypeptide.find('{http://www.drugbank.ca}synonyms').findall('{http://www.drugbank.ca}synonym')],
                'amino_acid_sequence': polypeptide.find('{http://www.drugbank.ca}amino-acid-sequence').text,
                'gene_sequence': polypeptide.find('{http://www.drugbank.ca}gene-sequence').text,
                'pfams': [{
                    'identifier': pfam.find('{http://www.drugbank.ca}identifier').text,
                    'name': pfam.find('{http://www.drugbank.ca}name').text
                } for pfam in polypeptide.find('{http://www.drugbank.ca}pfams').findall('{http://www.drugbank.ca}pfam')],
                'go_classifiers': [{
                    'category': go.find('{http://www.drugbank.ca}category').text,
                    'description': go.find('{http://www.drugbank.ca}description').text
                } for go in polypeptide.find('{http://www.drugbank.ca}go-classifiers').findall('{http://www.drugbank.ca}go-classifier')]
            } for polypeptide in target.findall('{http://www.drugbank.ca}polypeptide')]
        } for target in targets.findall('{http://www.drugbank.ca}target')[:max_samples]]
    
    # Enzymes
    enzymes = drug_element.find('{http://www.drugbank.ca}enzymes')
    if enzymes is not None:
        drug_data['enzymes'] = [{
            'id': enzyme.find('{http://www.drugbank.ca}id').text,
            'name': enzyme.find('{http://www.drugbank.ca}name').text,
            'organism': enzyme.find('{http://www.drugbank.ca}organism').text,
            'inhibition_strength': enzyme.find('{http://www.drugbank.ca}inhibition-strength').text,
            'induction_strength': enzyme.find('{http://www.drugbank.ca}induction-strength').text
        } for enzyme in enzymes.findall('{http://www.drugbank.ca}enzyme')[:max_samples]]

    # Carriers
    carriers = drug_element.find('{http://www.drugbank.ca}carriers')
    if carriers is not None:
        drug_data['carriers'] = [{
            'id': carrier.find('{http://www.drugbank.ca}id').text,
            'name': carrier.find('{http://www.drugbank.ca}name').text,
            'organism': carrier.find('{http://www.drugbank.ca}organism').text
        } for carrier in carriers.findall('{http://www.drugbank.ca}carrier')[:max_samples]]

    # Transporters
    transporters = drug_element.find('{http://www.drugbank.ca}transporters')
    if transporters is not None:
        drug_data['transporters'] = [{
            'id': transporter.find('{http://www.drugbank.ca}id').text,
            'name': transporter.find('{http://www.drugbank.ca}name').text,
            'organism': transporter.find('{http://www.drugbank.ca}organism').text
        } for transporter in transporters.findall('{http://www.drugbank.ca}transporter')[:max_samples]]

    return drug_data

# Main Function
def main(xml_file, pickle_file, max_samples=100000000):
    # Parsing XML
    tree = ET.parse(xml_file)
    root = tree.getroot()

    # Data extraction
    drugs = []
    statistics = {
        'total_drugs': 0,
        'total_salts': 0,
        'total_synonyms': 0,
        'total_products': 0,
        'total_international_brands': 0,
        'total_mixtures': 0,
        'total_packagers': 0,
        'total_manufacturers': 0,
        'total_prices': 0,
        'total_categories': 0,
        'total_affected_organisms': 0,
        'total_dosages': 0,
        'total_atc_codes': 0,
        'total_ahfs_codes': 0,
        'total_pdb_entries': 0,
        'total_targets': 0,
        'total_enzymes': 0,
        'total_carriers': 0,
        'total_transporters': 0,
        'total_drug_interactions': 0,
        'total_snp_effects': 0,
        'total_snp_adverse_drug_reactions': 0,
        'total_reactions': 0,
        'total_pathways': 0,
        'total_fasta':0,
        'total_food_interactions':0
    }

    drugs_elements = root.findall('{http://www.drugbank.ca}drug')[:max_samples]
    total_drugs = len(drugs_elements)  # Total number of drugs

    print("===================================")
    print("Total drugs to process", total_drugs)
    print("===================================")

    for drug in drugs_elements:
        drug_data = extract_drug_data(drug,max_samples=100000000)
        drugs.append(drug_data)

        # Statistics update
        statistics['total_drugs'] += 1
        statistics['total_fasta'] += 1 if drug_data.get('fasta_sequence') is not None else 0
        statistics['total_salts'] += len(drug_data.get('salts', []))
        statistics['total_synonyms'] += len(drug_data.get('synonyms', []))
        statistics['total_products'] += len(drug_data.get('products', []))
        statistics['total_international_brands'] += len(drug_data.get('international_brands', []))
        statistics['total_mixtures'] += len(drug_data.get('mixtures', []))
        statistics['total_packagers'] += len(drug_data.get('packagers', []))
        statistics['total_manufacturers'] += len(drug_data.get('manufacturers', []))
        statistics['total_prices'] += len(drug_data.get('prices', []))
        statistics['total_categories'] += len(drug_data.get('categories', []))
        statistics['total_affected_organisms'] += len(drug_data.get('affected_organisms', []))
        statistics['total_dosages'] += len(drug_data.get('dosages', []))
        statistics['total_atc_codes'] += len(drug_data.get('atc_codes', []))
        statistics['total_ahfs_codes'] += len(drug_data.get('ahfs_codes', []))
        statistics['total_pdb_entries'] += len(drug_data.get('pdb_entries', []))
        statistics['total_targets'] += len(drug_data.get('targets', []))
        statistics['total_enzymes'] += len(drug_data.get('enzymes', []))
        statistics['total_carriers'] += len(drug_data.get('carriers', []))
        statistics['total_transporters'] += len(drug_data.get('transporters', []))
        statistics['total_drug_interactions'] += len(drug_data.get('drug_interactions', []))
        statistics['total_snp_effects'] += len(drug_data.get('snp_effects', []))
        statistics['total_snp_adverse_drug_reactions'] += len(drug_data.get('snp_adverse_drug_reactions', []))
        statistics['total_reactions'] += len(drug_data.get('reactions', []))
        statistics['total_pathways'] += len(drug_data.get('pathways', []))
        statistics['total_food_interactions'] += len(drug_data.get('food_interactions',[]))
        if statistics['total_drugs'] % 1000 == 0:
            print(statistics['total_drugs'], "out of", total_drugs)
    # Pickle Serialization 
    with open(pickle_file, 'wb') as f:
        pickle.dump(drugs, f)

    # Print Statistics
    print("Statistics:")
    for key, value in statistics.items():
        print(f"{key}: {value}")

# Main execution
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Data Extraction from XML.')
    parser.add_argument('--xml_file', type=str, required=False,  default='../full database.xml', help='Drugbank XML file')
    parser.add_argument('--pickle_file', type=str, required=True, default='drugs_data.pkl', help='Pickle output file')
    parser.add_argument('--max_samples', type=int, default=100000000, help='Number of samples to extract (default: 100000000)')
    
    args = parser.parse_args()
    
    xml_file = args.xml_file  
    pickle_file = args.pickle_file  
    main(xml_file, pickle_file, args.max_samples)