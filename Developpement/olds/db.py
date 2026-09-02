"""
Base de données SQLite pour la composante saisonnière S_t (STL, section 3.2.2 /
4.4.3 du mémoire).

Cette table est IMMUABLE : elle est peuplée une seule fois (au premier lancement,
lors de la décomposition STL du Module 3) puis uniquement lue en lecture seule
pour reconstituer une prédiction (tendance_LSTM + S_t). Elle n'est pas exposée
dans l'interface : pas d'ajout, de modification ni de visualisation manuelle.
"""
import sqlite3
from pathlib import Path

DB_PATH = Path(__file__).parent / "seasonality.db"


def get_conn():
    return sqlite3.connect(DB_PATH, check_same_thread=False)


def init_db():
    conn = get_conn()
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS seasonality (
            date INTEGER NOT NULL,
            heure INTEGER NOT NULL,
            s_temp REAL,
            s_do REAL,
            s_ph REAL,
            s_turbidity REAL,
            PRIMARY KEY (date, heure)
        )
        """
    )
    conn.commit()
    conn.close()


def is_populated() -> bool:
    conn = get_conn()
    n = conn.execute("SELECT COUNT(*) FROM seasonality").fetchone()[0]
    conn.close()
    return n > 0


def populate_from_stl(seasonal_df):
    """Peuple la table une seule fois (no-op si déjà peuplée). Aucune mise à
    jour n'est faite ensuite : la saisonnalité est considérée figée une fois
    calculée par la décomposition STL."""
    if is_populated():
        return
    conn = get_conn()
    conn.executemany(
        "INSERT OR IGNORE INTO seasonality (date, heure, s_temp, s_do, s_ph, s_turbidity) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        seasonal_df[["date", "heure", "s_temp", "s_do", "s_ph", "s_turbidity"]].itertuples(
            index=False, name=None
        ),
    )
    conn.commit()
    conn.close()


def lookup(date: int, heure: int):
    """Lecture seule : renvoie (s_temp, s_do, s_ph, s_turbidity) ou None."""
    conn = get_conn()
    row = conn.execute(
        "SELECT s_temp, s_do, s_ph, s_turbidity FROM seasonality WHERE date=? AND heure=?",
        (int(date), int(heure)),
    ).fetchone()
    conn.close()
    return row
