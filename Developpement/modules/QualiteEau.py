import pickle
import os
from pathlib import Path
# 1. Trouve le dossier où se trouve le script Streamlit actuel
CURRENT_DIR = Path(__file__).parent
BASE_DIR = CURRENT_DIR.parent  # Remonte d'un niveau pour atteindre le dossier "modules"
MODEL_DIR = BASE_DIR / "models"  # Chemin vers le dossier "models"
class QualiteEau:
    # Donnée d'entrée : PH, TEMP, DO, TURBIDITY
    def __init__(self):
        print("Chargement Du modèle de Classification d'Eau ... ")
        with open(os.path.join(MODEL_DIR, "qualite_eau_xgboost.pkl"), "rb") as file:
            model = pickle.load(file)
        print("Chargement avec Succes ... ")
        self.model = model
    
    def predict(self,ph, temp, do, turbidity):
        
        input_data = [[ph, temp, do, turbidity]]
        prediction = self.model.predict(input_data)
        return prediction[0]

        