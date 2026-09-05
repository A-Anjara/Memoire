import streamlit as st
import pandas as pd
from datetime import datetime, timedelta
from modules.DensiteTilapia import DensiteTilapia
from modules.QualiteEau import QualiteEau
from modules.PredictionEau import PredictionEau
import math


densiteTilapia = DensiteTilapia()
qualiteEau = QualiteEau()
predictionEau = PredictionEau()
st.set_page_config(page_title="Système Rizipisciculture Intelligent", layout="wide")

st.title("Système Intelligent de Rizipisciculture")
st.caption(
    """
    Plateforme d'aide à la décision dédiée au suivi de la production
    rizipiscicole. Analysez les conditions d'élevage, évaluez la qualité
    de l'eau et anticipez son évolution à court terme.
    """
)


tab_m1, tab_m2, tab_m3 = st.tabs(
    [
        "Densité Tilapia",
        "Qualité actuelle de l'eau",
        "Anticipation de la qualité d'eau H+1",
    ]
)

# =====================================================================================
# MODULE 1 — Régression : densité de Tilapia
# =====================================================================================
with tab_m1:
    
    st.caption("""
    Estimez la densité de tilapia à partir des caractéristiques
            de la parcelle et des conditions actuelles d'élevage.
    """)
    c1, c2, c3 = st.columns(3)
    texture = c1.selectbox("Texture du sol", list(densiteTilapia.texture_sol_encoder.keys()))
    couleur = c2.selectbox("Couleur du sol", list(densiteTilapia.couleur_sol_encoder.keys()))
    irrigation = c3.selectbox("Type d'irrigation", list(densiteTilapia.type_irrigation_encoder.keys()))
    c4, c5, c6 = st.columns(3)
    hauteur = c4.slider("Hauteur d'eau (cm)", 5, 20, 12)
    engrais = c5.selectbox("Niveau d'engrais", list(densiteTilapia.niveaux_engrais_encoder.keys()))
    temperature = c6.selectbox(
        "Température actuelle", list(densiteTilapia.temperature_encoder.keys())
    )
    if st.button("Prédire la densité"):
        pred = densiteTilapia.predict(texture,couleur,irrigation,hauteur,engrais,temperature)
        st.metric("Densité de Tilapia estimée", f"{math.floor(pred)} / are")


# =====================================================================================
# MODULE 2 — Classification : qualité actuelle de l'eau
# =====================================================================================
with tab_m2:
    
    st.caption("""
    Analysez les principaux paramètres physico-chimiques afin
            d'identifier l'état actuel de l'eau dans la parcelle.
            """)
            
    c1, c2, c3, c4 = st.columns(4)
    ph = c1.number_input("PH", 0.0, 14.0, 6.5,0.5)
    temp = c2.number_input("Température (°C)", 0.0, 60.0, 27.0,1.0)
    do = c3.number_input("Oxygène dissous (mg/L)", 0.0, 30.0, 7.0,0.5)
    turbidity = c4.number_input("Turbidité (NTU)", 0.0, 100.0, 30.0,1.0)
    
    if st.button("Prédire la qualité"):
        pred = qualiteEau.predict(ph, temp, do, turbidity)
        label = "Bonne qualité" if pred == 1 else "Mauvaise qualité"
        st.metric("Qualité de l'eau estimée", label)
   


with tab_m3:
    st.caption("""
    Analysez l'évolution récente des paramètres de l'eau à partir
            des mesures H-2, H-1 et H afin d'anticiper son état à H+1.
    """)
# --- ROW 1: Hour H-2 ---
    st.markdown("#### 2 Heures avant (H - 2)")
    c1, c2, c3, c4 = st.columns(4)
    ph_h2 = c1.number_input("PH", 0.0, 14.0, 6.5, 0.5, key="ph_h2")
    temp_h2 = c2.number_input("Température (°C)", 0.0, 60.0, 27.0, 1.0, key="temp_h2")
    do_h2 = c3.number_input("Oxygène dissous (mg/L)", 0.0, 30.0, 7.0, 0.5, key="do_h2")
    turbidity_h2 = c4.number_input("Turbidité (NTU)", 0.0, 100.0, 30.0, 1.0, key="turb_h2")

    st.markdown("---")  # Visual separator line

    # --- ROW 2: Hour H-1 ---
    st.markdown("#### 1 Heure avant (H - 1)")
    c1, c2, c3, c4 = st.columns(4)
    ph_h1 = c1.number_input("PH", 0.0, 14.0, 6.5, 0.5, key="ph_h1")
    temp_h1 = c2.number_input("Température (°C)", 0.0, 60.0, 27.0, 1.0, key="temp_h1")
    do_h1 = c3.number_input("Oxygène dissous (mg/L)", 0.0, 30.0, 7.0, 0.5, key="do_h1")
    turbidity_h1 = c4.number_input("Turbidité (NTU)", 0.0, 100.0, 30.0, 1.0, key="turb_h1")

    st.markdown("---")

    # --- ROW 3: Current Hour H ---
    st.markdown("#### Heure Actuelle")
    c1, c2, c3, c4 = st.columns(4)
    ph_h = c1.number_input("PH", 0.0, 14.0, 6.5, 0.5, key="ph_h")
    temp_h = c2.number_input("Température (°C)", 0.0, 60.0, 27.0, 1.0, key="temp_h")
    do_h = c3.number_input("Oxygène dissous (mg/L)", 0.0, 30.0, 7.0, 0.5, key="do_h")
    turbidity_h = c4.number_input("Turbidité (NTU)", 0.0, 100.0, 30.0, 1.0, key="turb_h")

    


    
    if st.button("Anticiper"):
        ph_list = [ph_h2, ph_h1, ph_h]
        temp_list = [temp_h2, temp_h1, temp_h]
        do_list = [do_h2, do_h1, do_h]
        turbidity_list = [turbidity_h2, turbidity_h1, turbidity]
        # Calcul de l'heure cible H+1
        now = datetime.now()
        

        # Arrondir à l'heure
        current_hour = datetime(now.year,now.month,now.day,now.hour)
        first_year = datetime(now.year,1,1,0)

        delta = current_hour - first_year
        hours = delta.total_seconds()//3600

        pred = predictionEau.predict(ph_list, temp_list, do_list, turbidity_list, hours)
        st.metric("PH", f"{pred['PH']:.2f}")
        st.markdown("- - -")
        st.metric("Température", f"{pred['TEMP']:.2f}")
        st.markdown("- - -")
        st.metric("Oxygène dissous", f"{pred['DO']:.2f}")
        st.markdown("- - -")
        st.metric("Turbidité", f"{pred['TURBIDITY']:.2f}")
        

        




