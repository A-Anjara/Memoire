import streamlit as st
import sqlite3
import pandas as pd
import datetime
import random

# --- CONFIGURATION DE LA PAGE ---
st.set_page_config(
    page_title="RiziPisci-Smart - Madagascar",
    page_icon="🐟",
    layout="wide"
)

# --- INITIALISATION DE LA BASE DE DONNÉES SQLITE ---
def init_db():
    conn = sqlite3.connect('rizipisciculture.db')
    c = conn.cursor()
    # Table Module 1 : Caractéristiques du sol et densité
    c.execute('''CREATE TABLE IF NOT EXISTS module1_sol (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    date TEXT,
                    texture_sol TEXT,
                    couleur_sol TEXT,
                    type_irrigation TEXT,
                    hauteur_eau REAL,
                    niveau_engrais TEXT,
                    temp_actuelle REAL,
                    densite_predite REAL)''')
    
    # Table Modules 2 & 3 : Paramètres physico-chimiques et qualité de l'eau
    c.execute('''CREATE TABLE IF NOT EXISTS module2_3_eau (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    date TEXT,
                    time TEXT,
                    ph REAL,
                    temp REAL,
                    do REAL,
                    turbidity REAL,
                    qualite_actuelle TEXT,
                    ph_h1 REAL,
                    temp_h1 REAL,
                    do_h1 REAL,
                    turbidity_h1 REAL)''')
    conn.commit()
    conn.close()

init_db()

# --- FONCTIONS DE SIMULATION DES MODÈLES ML (Heuristiques basées sur le mémoire) ---
def simuler_module1(texture, couleur, irrigation, hauteur, engrais, temp):
    # Simulation d'une régression (Random Forest / XGBoost)
    score = 15.0
    if texture in ["Argileux", "Limoneux"]: score += 5.0
    if couleur in ["Foncé (Riche)"]: score += 4.0
    if irrigation == "Continu": score += 3.0
    if 10 <= hauteur <= 25: score += 5.0
    if engrais == "Fort": score += 2.0
    if 24 <= temp <= 30: score += 4.0
    return round(score + random.uniform(-1, 1), 2)

def simuler_module2_classification(ph, temp, do, turbidity):
    # Modèle basé sur les seuils critiques du Tilapia (Simule Random Forest)
    # 1 = Bonne qualité, 0 = Mauvaise qualité
    if (6.5 <= ph <= 8.5) and (22 <= temp <= 32) and (do >= 4.0) and (turbidity <= 40):
        return "Bonne qualité"
    else:
        return "Mauvaise qualité"

def simuler_module3_lstm(ph, temp, do, turbidity):
    # Simule l'évolution de la tendance à H+1 (Modèle LSTM)
    ph_next = round(ph + random.uniform(-0.2, 0.2), 2)
    temp_next = round(temp + random.uniform(-0.8, 0.8), 2)
    do_next = round(max(0.0, do + random.uniform(-0.5, 0.4)), 2)
    turb_next = round(max(0.0, turbidity + random.uniform(-3, 5)), 2)
    return ph_next, temp_next, do_next, turb_next

# --- INTERFACE UTILISATEUR (STREAMLIT) ---
st.title("🐟 RiziPisci-Smart Dashboard")
st.caption("Système d'Aide à la Décision Analytique pour la Rizipisciculture à Madagascar (INSI)")

# Menu de navigation latérale
menu = st.sidebar.radio(
    "Navigation du Système",
    ["Vue d'ensemble", "Module 1 : Densité du Tilapia", "Modules 2 & 3 : Analyse de l'Eau", "Historique SQLite"]
)

# --- 1. VUE D'ENSEMBLE ---
if menu == "Vue d'ensemble":
    st.header("📋 Présentation du Projet")
    st.write("""
    Cette application est une implémentation interactive des trois modules d'apprentissage automatique 
    développés pour optimiser la production conjointe de riz et de tilapias sur les Hautes-Terres malgaches.
    """)
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.info("📊 **Module 1**\n\nEstime la **densité idéale de poissons** (par are) basée sur la nature et les nutriments du sol rizicole.")
    with col2:
        st.success("🛡️ **Module 2**\n\nClassifie instantanément si la **qualité actuelle de l'eau** est saine ou dangereuse pour le cheptel.")
    with col3:
        st.warning("🔮 **Module 3**\n\nAnticipe à **H+1** les dégradations physico-chimiques via décomposition temporelle (STL+LSTM).")

# --- 2. MODULE 1 ---
elif menu == "Module 1 : Densité du Tilapia":
    st.header("🌾 Module 1 : Estimation de la Densité (Régression)")
    st.subheader("Saisie des caractéristiques de la parcelle rizicole")
    
    with st.form("form_module1"):
        col1, col2 = st.columns(2)
        with col1:
            texture = col1.selectbox("Texture du sol", ["Sableux (Filtrant)", "Limoneux", "Argileux (Retenant)"])
            couleur = col1.selectbox("Couleur du sol", ["Clair", "Intermédiaire", "Foncé (Riche)"])
            irrigation = col1.selectbox("Type d'irrigation", ["Pluvial (Ponctuel)", "Intermittent", "Continu"])
        with col2:
            hauteur_eau = col2.number_input("Hauteur d'eau dans la rizière (cm)", min_value=0.0, max_value=100.0, value=15.0)
            engrais = col2.selectbox("Niveau de fertilisation", ["Faible", "Moyen", "Fort"])
            temp_actuelle = col2.slider("Température Ambiante actuelle (°C)", 15.0, 40.0, 25.0)
            
        submit_m1 = st.form_submit_button("Calculer la densité recommandée")
        
    if submit_m1:
        res_densite = simuler_module1(texture, couleur, irrigation, hauteur_eau, engrais, temp_actuelle)
        st.metric(label="🔹 Densité de Tilapia Recommandée", value=f"{res_densite} poissons / are")
        
        # Sauvegarde SQLite
        conn = sqlite3.connect('rizipisciculture.db')
        c = conn.cursor()
        c.execute("INSERT INTO module1_sol (date, texture_sol, couleur_sol, type_irrigation, hauteur_eau, niveau_engrais, temp_actuelle, densite_predite) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                  (str(datetime.date.today()), texture, couleur, irrigation, hauteur_eau, engrais, temp_actuelle, res_densite))
        conn.commit()
        conn.close()
        st.success("✅ Résultat enregistré avec succès dans la base SQLite !")

# --- 3. MODULES 2 & 3 ---
elif menu == "Modules 2 & 3 : Analyse de l'Eau":
    st.header("💧 Modules 2 & 3 : État Actuel & Anticipation Temporelle")
    
    with st.form("form_eau"):
        col1, col2 = st.columns(2)
        with col1:
            ph = col1.number_input("pH de l'eau (Actuel)", min_value=0.0, max_value=14.0, value=7.2)
            temp = col1.number_input("Température de l'eau (°C) (Actuel)", min_value=0.0, max_value=50.0, value=26.5)
        with col2:
            do = col2.number_input("Oxygène Dissous (mg/L) (Actuel)", min_value=0.0, max_value=20.0, value=5.5)
            turbidity = col2.number_input("Turbidité (NTU) (Actuel)", min_value=0.0, max_value=500.0, value=25.0)
            
        submit_eau = st.form_submit_button("Lancer les analyses instantanées et prédictives")
        
    if submit_eau:
        # Module 2 : Classification
        statut_qualite = simuler_module2_classification(ph, temp, do, turbidity)
        
        # Module 3 : LSTM à H+1
        ph_h1, temp_h1, do_h1, turbidity_h1 = simuler_module3_lstm(ph, temp, do, turbidity)
        statut_qualite_h1 = simuler_module2_classification(ph_h1, temp_h1, do_h1, turbidity_h1)
        
        # Affichage des Résultats
        st.subheader("🛡️ Résultat Module 2 : Diagnostic Immédiat")
        if statut_qualite == "Bonne qualité":
            st.success(f"L'eau est actuellement de : **{statut_qualite}**")
        else:
            st.error(f"⚠️ Alerte : L'eau est actuellement de : **{statut_qualite}**")
            
        st.subheader("🔮 Résultat Module 3 : Anticipation à H+1 (Tendance Future)")
        
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("pH prévisionnel", f"{ph_h1}", f"{round(ph_h1-ph, 2)}")
        c2.metric("Température prévisionnelle", f"{temp_h1} °C", f"{round(temp_h1-temp, 2)} °C")
        c3.metric("Oxygène dissous prévisionnel", f"{do_h1} mg/L", f"{round(do_h1-do, 2)} mg/L")
        c4.metric("Turbidité prévisionnelle", f"{turbidity_h1} NTU", f"{round(turbidity_h1-turbidity, 2)} NTU")
        
        if statut_qualite_h1 == "Mauvaise qualité":
            st.warning("🚨 Attention : Risque de dégradation de la qualité de l'eau détecté dans l'heure qui vient !")
        else:
            st.success("🟢 Aucune anomalie critique attendue à H+1.")
            
        # Sauvegarde SQLite
        conn = sqlite3.connect('rizipisciculture.db')
        c = conn.cursor()
        now = datetime.datetime.now()
        c.execute("""INSERT INTO module2_3_eau 
                  (date, time, ph, temp, do, turbidity, qualite_actuelle, ph_h1, temp_h1, do_h1, turbidity_h1) 
                  VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                  (str(now.date()), now.strftime("%H:%M:%S"), ph, temp, do, turbidity, statut_qualite, ph_h1, temp_h1, do_h1, turbidity_h1))
        conn.commit()
        conn.close()
        st.success("✅ Mesures et projections mémorisées dans la base SQLite.")

# --- 4. HISTORIQUE ---
elif menu == "Historique SQLite":
    st.header("🗄️ Données Stockées en Base de Données")
    
    conn = sqlite3.connect('rizipisciculture.db')
    
    st.subheader("📈 Suivi du Module 1 (Sol & Densité)")
    df1 = pd.read_sql_query("SELECT * FROM module1_sol ORDER BY id DESC", conn)
    if not df1.empty:
        st.dataframe(df1, use_container_width=True)
    else:
        st.info("Aucun enregistrement pour le moment.")
        
    st.subheader("📊 Suivi des Modules 2 & 3 (Qualité & Prévisions Horaires)")
    df2 = pd.read_sql_query("SELECT * FROM module2_3_eau ORDER BY id DESC", conn)
    if not df2.empty:
        st.dataframe(df2, use_container_width=True)
    else:
        st.info("Aucun enregistrement pour le moment.")
        
    conn.close()
