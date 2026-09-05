import pickle
import os
class DensiteTilapia:
    # Variable d'entrée : Texture_sol,	Couleur_sol, Type_irrigation, Hauteur_Eau, Niveau_Engrais, Temperature_Actuelle
    def __init__(self):
        self.temperature_encoder = {
            "TresFroid":0,
            "Froid":1,
            "Humide":2,
            "Chaud":3,
            "TresChaud":4,
        }
        self.niveaux_engrais_encoder = {
            "Nul":0,
            "Moyen":1,
            "Fort":2
        }
        self.texture_sol_encoder = {
            "Sableux":0,
            "Limoneux":1,
            "Argileux":2,
        }
        self.couleur_sol_encoder = {
            "Orange_Rouge":1,
            "Brun_Grisatre":2,
            "Noir":3
        }
        self.type_irrigation_encoder = {
            "Pluvial":0,
            "Continu":1
        }

        print("Chargement Du modèle de Prédiction de Densité Tilapia ... ")
        with open(os.path.join("models", "densite_tilapia_xgboost.pkl"), "rb") as file:
            model = pickle.load(file)
        print("Chargement avec Succes ... ")

        self.model = model

    def predict(self, texture, couleur , type_irrigation , hauteur_eau, engrais, temperature):
        encoded_features = [
            self.texture_sol_encoder[texture],
            self.couleur_sol_encoder[couleur],
            self.type_irrigation_encoder[type_irrigation],
            hauteur_eau,
            self.niveaux_engrais_encoder[engrais],
            self.temperature_encoder[temperature]
        ]
        prediction = self.model.predict([encoded_features])
        return prediction[0]
    