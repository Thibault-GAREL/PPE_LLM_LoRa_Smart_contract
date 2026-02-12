"""
csv_loader.py
Module pour charger et afficher les informations du dataset CSV
"""
import pandas as pd


def lire_csv(chemin_fichier):
    """
    Lit un fichier CSV et retourne un DataFrame.
    
    Args:
        chemin_fichier: Chemin vers le fichier CSV
        
    Returns:
        DataFrame pandas
    """
    try:
        df = pd.read_csv(chemin_fichier)
        return df
    except FileNotFoundError:
        print(f"❌ Erreur : Le fichier '{chemin_fichier}' n'existe pas.")
        raise
    except Exception as e:
        print(f"❌ Erreur lors de la lecture du CSV : {e}")
        raise


def afficher_info_dataset(df):
    """
    Affiche les informations sur le dataset.
    
    Args:
        df: DataFrame pandas
    """
    print(f"Nombre total de contrats : {len(df)}")
    print(f"Colonnes disponibles : {list(df.columns)}")
    
    if 'label_encoded' in df.columns:
        print(f"\nDistribution des labels :")
        print(df['label_encoded'].value_counts().sort_index())
    
    # Vérifier les valeurs manquantes
    missing = df.isnull().sum()
    if missing.any():
        print(f"\nValeurs manquantes :")
        print(missing[missing > 0])
    else:
        print(f"\n✅ Aucune valeur manquante")


if __name__ == "__main__":
    # Test du module
    import sys
    
    if len(sys.argv) > 1:
        fichier = sys.argv[1]
    else:
        fichier = "archive/SC_Vuln_8label.csv"
    
    print(f"Chargement de {fichier}...\n")
    df = lire_csv(fichier)
    afficher_info_dataset(df)
    
    # Afficher les premières lignes
    print(f"\nPremières lignes :")
    print(df.head())
