import sqlite3
from pathlib import Path
import pandas as pd

from pathlib import Path
# 1. Trouve le dossier où se trouve le script Streamlit actuel
CURRENT_DIR = Path(__file__).parent
BASE_DIR = CURRENT_DIR.parent  # Remonte d'un niveau pour atteindre le dossier "modules"

DB_PATH = BASE_DIR / "services" / "seasonality.db"
SEASONALITY_CSV_PATH = BASE_DIR / "services" / "seasonality.csv"

# df  = pd.read_csv(r"seasonality.csv")

def get_conn():
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    conn = get_conn()
    print("Creation Table ...")
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS seasonality (
            index_heure INTEGER PRIMARY KEY,
            s_temp REAL NOT NULL,
            s_do REAL NOT NULL,
            s_ph REAL NOT NULL,
            s_turbidity REAL NOT NULL
        );
        """
    )
    conn.commit()
    conn.close()
    print("Creation Avec succes ...")
    conn = get_conn()
    cursor = conn.cursor()
    df  = pd.read_csv(SEASONALITY_CSV_PATH)
    df.to_sql("seasonality", conn, if_exists="replace", index=False)
    print("Mise de saisonnalité ... ")
    
    conn.commit()
    cursor.close()
    conn.close()
    



def get_saisonnalite(nom_composant_sql: str, index_heure : int) -> list[float]:
    conn = get_conn()
    cursor = conn.cursor()
    index_db = ((index_heure-1)%8759)+1
    cursor.execute(f"SELECT {nom_composant_sql} FROM seasonality WHERE index_heure = {index_db}")
    result = cursor.fetchone()
    cursor.close()
    conn.close()
    return dict(result)[nom_composant_sql]
