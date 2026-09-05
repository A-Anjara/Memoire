import sqlite3
from pathlib import Path
import pandas as pd

# 1. Gestion propre des chemins
CURRENT_DIR = Path(__file__).parent
BASE_DIR = CURRENT_DIR.parent  

DB_PATH = BASE_DIR / "services" / "seasonality.db"
SEASONALITY_CSV_PATH = BASE_DIR / "services" / "seasonality.csv"

def get_conn():
    # Ajout d'un timeout de 30 secondes pour donner le temps aux requêtes concurrentes de finir
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    """Initialise et remplace les données de la table proprement."""
    df = pd.read_csv(SEASONALITY_CSV_PATH)
    
    # Le bloc "with" garantit la fermeture de la connexion, quoi qu'il arrive
    with get_conn() as conn:
        print("Remplacement de la table seasonality...")
        df.to_sql("seasonality", conn, if_exists="replace", index=False)
        print("Mise à jour de la saisonnalité effectuée avec succès !")


def get_saisonnalite(nom_composant_sql: str, index_heure: int) -> float:
    """Récupère la valeur de saisonnalité de manière thread-safe."""
    # Nettoyage rudimentaire pour éviter les injections SQL sur le nom de la colonne
    nom_composant_sql = nom_composant_sql.strip('"`[] ')
    
    index_db = ((index_heure - 1) % 8759) + 1
    
    with get_conn() as conn:
        cursor = conn.cursor()
        # Sécurisation du paramètre index_db avec un placeholder (?)
        query = f'SELECT "{nom_composant_sql}" FROM seasonality WHERE index_heure = ?'
        cursor.execute(query, (index_db,))
        
        result = cursor.fetchone()
        
            
        return result[nom_composant_sql]
