import pickle
import os
class QualiteEau:
    # Donnée d'entrée : PH, TEMP, DO, TURBIDITY
    def __init__(self):
        print("Chargement Du modèle de Classification d'Eau ... ")
        with open(os.path.join("models", "qualite_eau_xgboost.pkl"), "rb") as file:
            model = pickle.load(file)
        print("Chargement avec Succes ... ")
        self.model = model
    
    def predict(self,ph, temp, do, turbidity):
        
        input_data = [[ph, temp, do, turbidity]]
        prediction = self.model.predict(input_data)
        return prediction[0]

        