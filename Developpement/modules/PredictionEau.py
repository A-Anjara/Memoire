import pickle 
import os
import numpy as np
from statsmodels.tsa.seasonal import STL
from services.db import get_saisonnalite

from pathlib import Path
#Trouve le dossier où se trouve le script Streamlit actuel
CURRENT_DIR = Path(__file__).parent
BASE_DIR = CURRENT_DIR.parent  # Remonte d'un niveau pour atteindre le dossier "modules"
MODEL_DIR = BASE_DIR / "models"  # Chemin vers le dossier "models"


class PredictionEau:
    VARS = ["PH", "TEMP", "DO", "TURBIDITY"]
    COL_MAP = {"PH": "s_ph", "TEMP": "s_temp", "DO": "s_do", "TURBIDITY": "s_turbidity"}
    # Donnée d'entrée : PH, TEMP, DO, TURBIDITY
    def __init__(self):
        print("Chargement Du modèle de Prédiction d'Eau ... ")
        self.model = {}
        self.scaler = {}
        print("here")
        with open(os.path.join(MODEL_DIR, "Timeseries_DL_PH_trend_LSTM_64.pkl"), "rb") as file:
            model = pickle.load(file)
        self.model["PH"] = model

        print("here")
        with open(os.path.join(MODEL_DIR, "Timeseries_DL_TEMP_trend_LSTM_64.pkl"), "rb") as file:
            model = pickle.load(file)
        self.model["TEMP"] = model

        print("here")
        with open(os.path.join(MODEL_DIR, "Timeseries_DL_DO_trend_LSTM_64.pkl"), "rb") as file:
            model = pickle.load(file)
        self.model["DO"] = model

        print("here")
        with open(os.path.join(MODEL_DIR, "Timeseries_DL_TURBIDITY_trend_LSTM_64.pkl"), "rb") as file:
            model = pickle.load(file)
        self.model["TURBIDITY"] = model

        print("here")
        with open(os.path.join(MODEL_DIR, "PH_trend_scaler.pkl"), "rb") as file:
            scaler = pickle.load(file)
        self.scaler["PH"] = scaler

        print("here")
        with open(os.path.join(MODEL_DIR, "TEMP_trend_scaler.pkl"), "rb") as file:
            scaler = pickle.load(file)
        self.scaler["TEMP"] = scaler

        print("here")
        with open(os.path.join(MODEL_DIR, "DO_trend_scaler.pkl"), "rb") as file:
            scaler = pickle.load(file)
        self.scaler["DO"] = scaler

        print("here")
        with open(os.path.join(MODEL_DIR, "TURBIDITY_trend_scaler.pkl"), "rb") as file:
            scaler = pickle.load(file)
        self.scaler["TURBIDITY"] = scaler
        

        print("Chargement avec Succes ... ")
    
    def predict(self, ph_list, temp_list, do_list, turbidity_list, hour_of_year_target):
        """
        ph_list, temp_list, etc. : list[3] (Lags H-2, H-1, H)
        hour_of_year_target : int (Index de 1 à ~8800 pour l'heure H+1 à prédire)
        """
        inputs = {
            "PH": ph_list,
            "TEMP": temp_list,
            "DO": do_list,
            "TURBIDITY": turbidity_list
        }
        
        predictions_finales = {}

        for var in self.VARS:
            # 1. Extraction de la tendance via la décomposition STL
            res = STL(inputs[var], period=3, robust=True).fit()
            tendance = res.trend
            
            # 2. Reshape en (3, 1) pour appliquer le Scaler sur les données brutes
            tendance_3_1 = np.array(tendance).reshape(3, 1)
            tendance_scaled = self.scaler[var].transform(tendance_3_1)
            
            # 3. Reshape en (1, 3, 1) [Samples, Timesteps, Features] pour l'entrée du modèle LSTM
            input_modele = tendance_scaled.reshape(1, 3, 1)
            
            # 4. Prédiction de la tendance scalée pour H+1
            pred_trend_scaled = self.model[var].predict(input_modele)
            
            # 5. Inversement du scaling pour retrouver la vraie valeur de la tendance
            # Note : On reshape le résultat en (1, 1) pour l'inverse_transform
            pred_trend = self.scaler[var].inverse_transform(pred_trend_scaled.reshape(1, 1))[0][0]
            
            # 6. Récupération de la saisonnalité depuis SQL via votre module (clé COL_MAP pour le nom du composant)
            nom_composant_sql = self.COL_MAP[var]
            saisonnalite_sql = get_saisonnalite(nom_composant_sql, hour_of_year_target)
            
            # 7. Reconstruction du signal final (Tendance H+1 + Saisonnalité H+1)
            predictions_finales[var] = pred_trend + saisonnalite_sql
            
        return predictions_finales
