import pickle 
import os

class PredictionEau:
    VARS = ["PH", "TEMP", "DO", "TURBIDITY"]
    COL_MAP = {"PH": "s_ph", "TEMP": "s_temp", "DO": "s_do", "TURBIDITY": "s_turbidity"}
    # Donnée d'entrée : PH, TEMP, DO, TURBIDITY
    def __init__(self):
        print("Chargement Du modèle de Prédiction d'Eau ... ")
        self.model = {}
        self.scaler = {}
        with open(os.path.join("models", "Timeseries_DL_PH_trend_LSTM_64.pkl"), "rb") as file:
            model = pickle.load(file)
        self.model["PH"] = model

        with open(os.path.join("models", "Timeseries_DL_TEMP_trend_LSTM_64.pkl"), "rb") as file:
            model = pickle.load(file)
        self.model["TEMP"] = model

        with open(os.path.join("models", "Timeseries_DL_DO_trend_LSTM_64.pkl"), "rb") as file:
            model = pickle.load(file)
        self.model["DO"] = model

        with open(os.path.join("models", "Timeseries_DL_TURBIDITY_trend_LSTM_64.pkl"), "rb") as file:
            model = pickle.load(file)
        self.model["TURBIDITY"] = model

        with open(os.path.join("models", "PH_trend_scaler.pkl"), "rb") as file:
            scaler = pickle.load(file)
        self.scaler["PH"] = scaler

        with open(os.path.join("models", "TEMP_trend_scaler.pkl"), "rb") as file:
            scaler = pickle.load(file)
        self.scaler["TEMP"] = scaler

        with open(os.path.join("models", "DO_trend_scaler.pkl"), "rb") as file:
            scaler = pickle.load(file)
        self.scaler["DO"] = scaler

        with open(os.path.join("models", "TURBIDITY_trend_scaler.pkl"), "rb") as file:
            scaler = pickle.load(file)
        self.scaler["TURBIDITY"] = scaler
        

        print("Chargement avec Succes ... ")
    
    def predict(self,ph, temp, do, turbidity):
        
        input_data = [[ph, temp, do, turbidity]]
        prediction = self.model.predict(input_data)
        return prediction[0]