import pickle
import csv
import argparse

def main(pickle_file):
    # Caricare i dati dal file pickle
    with open(pickle_file, 'rb') as f:
        drugs = pickle.load(f)

    # Inizializzare le statistiche
    classification_counts = {
        'total_direct_parent': 0,
        'total_kingdom': 0,
        'total_superclass': 0,
        'total_class': 0,
        'total_subclass': 0,
        'total_alternative_parents': 0,
        'total_substituents': 0,
        'no_direct_parent': 0,
        'no_kingdom': 0,
        'no_superclass': 0,
        'no_class': 0,
        'no_subclass': 0,
        'none_of_above': 0,
        'at_least_one': 0,
        'no_alternative_parents': 0,
        'no_substituents': 0,
        'at_least_one_alt_parent_or_subst': 0
    }

    # Insiemi per i valori unici
    unique_direct_parent = set()
    unique_kingdom = set()
    unique_superclass = set()
    unique_class = set()
    unique_subclass = set()
    unique_alternative_parents = set()
    unique_substituents = set()

    # Elaborare ciascun farmaco
    for drug in drugs:
        classification = drug.get('classification', {})

        # Contare i valori non nulli e raccogliere i valori unici
        has_any = False

        if 'direct_parent' in classification and classification['direct_parent']:
            classification_counts['total_direct_parent'] += 1
            unique_direct_parent.add(classification['direct_parent'])
            has_any = True
        else:
            classification_counts['no_direct_parent'] += 1

        if 'kingdom' in classification and classification['kingdom']:
            classification_counts['total_kingdom'] += 1
            unique_kingdom.add(classification['kingdom'])
            has_any = True
        else:
            classification_counts['no_kingdom'] += 1

        if 'superclass' in classification and classification['superclass']:
            classification_counts['total_superclass'] += 1
            unique_superclass.add(classification['superclass'])
            has_any = True
        else:
            classification_counts['no_superclass'] += 1

        if 'class' in classification and classification['class']:
            classification_counts['total_class'] += 1
            unique_class.add(classification['class'])
            has_any = True
        else:
            classification_counts['no_class'] += 1

        if 'subclass' in classification and classification['subclass']:
            classification_counts['total_subclass'] += 1
            unique_subclass.add(classification['subclass'])
            has_any = True
        else:
            classification_counts['no_subclass'] += 1

        if not has_any:
            classification_counts['none_of_above'] += 1
        else:
            classification_counts['at_least_one'] += 1

        alt_parents = classification.get('alternative_parents', [])
        if alt_parents:
            classification_counts['total_alternative_parents'] += len(alt_parents)
            unique_alternative_parents.update(alt_parents)
        else:
            classification_counts['no_alternative_parents'] += 1

        substituents = classification.get('substituents', [])
        if substituents:
            classification_counts['total_substituents'] += len(substituents)
            unique_substituents.update(substituents)
        else:
            classification_counts['no_substituents'] += 1

        if alt_parents or substituents:
            classification_counts['at_least_one_alt_parent_or_subst'] += 1

    # Stampare le statistiche
    print("Statistics:")
    for key, value in classification_counts.items():
        print(f"{key}: {value}")

    # Funzione per scrivere i set unici nei file CSV
    def write_unique_to_csv(filename, unique_set):
        with open(filename, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            for item in sorted(unique_set):
                writer.writerow([item])

    # Scrivere i valori unici nei file CSV
    write_unique_to_csv('unique_direct_parent.csv', unique_direct_parent)
    write_unique_to_csv('unique_kingdom.csv', unique_kingdom)
    write_unique_to_csv('unique_superclass.csv', unique_superclass)
    write_unique_to_csv('unique_class.csv', unique_class)
    write_unique_to_csv('unique_subclass.csv', unique_subclass)
    write_unique_to_csv('unique_alternative_parents.csv', unique_alternative_parents)
    write_unique_to_csv('unique_substituents.csv', unique_substituents)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Estrai classificazioni dai file pickle di farmaci.')
    parser.add_argument('pickle_file', type=str, help='Il file pickle contenente i dati dei farmaci.')
    args = parser.parse_args()
    main(args.pickle_file)