import streamlit as st
import pandas as pd

import db
import module1
import module2
import module3

st.set_page_config(page_title="Système Rizipisciculture Intelligent", layout="wide", page_icon="🐟")

st.title("🌾🐟 Système Intelligent de Rizipisciculture")
st.caption(
    "Démonstrateur des 3 modules décrits dans le mémoire (INSI). ⚠️ Aucun jeu de données réel "
    "n'a été fourni : les modèles sont entraînés sur des données **synthétiques**, générées pour "
    "reproduire les distributions décrites en section 4.3, à des fins de démonstration du pipeline."
)

db.init_db()

tab_m1, tab_m2, tab_m3 = st.tabs(
    [
        "🌱 Module 1 — Densité Tilapia",
        "💧 Module 2 — Qualité actuelle de l'eau",
        "⏱️ Module 3 — Anticipation H+1",
    ]
)

# =====================================================================================
# MODULE 1 — Régression : densité de Tilapia
# =====================================================================================
with tab_m1:
    st.header("Module 1 — Estimation de la densité de Tilapia")
    st.caption(
        "Régression : Régression Linéaire, Arbre de Décision, Random Forest, SVR, XGBoost — "
        "optimisés par GridSearch (section 5.1)."
    )

    if st.button("🚀 Entraîner / ré-entraîner les modèles du Module 1"):
        with st.spinner("Génération des données et GridSearch en cours..."):
            results, models, scaler = module1.train_all_models()
        st.session_state["m1_results"] = results
        st.session_state["m1_models"] = models
        st.session_state["m1_scaler"] = scaler
        st.success("Modèles du Module 1 entraînés.")

    if "m1_results" in st.session_state:
        res_df = pd.DataFrame(st.session_state["m1_results"]).T
        st.dataframe(res_df[["cv_r2", "RMSE", "MAE", "MAPE", "R2"]], use_container_width=True)

        st.subheader("Prédiction")
        c1, c2, c3 = st.columns(3)
        texture = c1.selectbox("Texture du sol", ["Sableux", "Limoneux", "Argileux"])
        couleur = c2.selectbox("Couleur du sol", ["Orange_Rouge", "Brun_Grisatre", "Noir"])
        irrigation = c3.selectbox("Type d'irrigation", ["Pluvial", "Continu"])
        c4, c5, c6 = st.columns(3)
        hauteur = c4.slider("Hauteur d'eau (cm)", 5, 20, 12)
        engrais = c5.selectbox("Niveau d'engrais", ["Nul", "Moyen", "Fort"])
        temperature = c6.selectbox(
            "Température actuelle", ["TresFroid", "Froid", "Humide", "Chaud", "TresChaud"]
        )
        model_choice = st.selectbox(
            "Modèle à utiliser", list(st.session_state["m1_models"].keys()), index=4
        )

        if st.button("Prédire la densité"):
            pred = module1.predict_density(
                st.session_state["m1_models"],
                st.session_state["m1_scaler"],
                model_choice,
                texture,
                couleur,
                irrigation,
                hauteur,
                engrais,
                temperature,
            )
            st.metric("Densité de Tilapia estimée", f"{pred:.1f} / are")
    else:
        st.info("Cliquez sur le bouton ci-dessus pour entraîner les modèles.")

# =====================================================================================
# MODULE 2 — Classification : qualité actuelle de l'eau
# =====================================================================================
with tab_m2:
    st.header("Module 2 — Classification de l'état actuel de l'eau")
    st.caption(
        "Classification binaire : Régression Logistique, Arbre de Décision, Random Forest, SVC, "
        "XGBoost — optimisés par GridSearch, jeu de données ré-équilibré (section 5.2)."
    )

    if st.button("🚀 Entraîner / ré-entraîner les modèles du Module 2"):
        with st.spinner("Génération des données, ré-équilibrage et GridSearch en cours..."):
            results, models, scaler = module2.train_all_models()
        st.session_state["m2_results"] = results
        st.session_state["m2_models"] = models
        st.session_state["m2_scaler"] = scaler
        st.success("Modèles du Module 2 entraînés.")

    if "m2_results" in st.session_state:
        res_df = pd.DataFrame(st.session_state["m2_results"]).T
        st.dataframe(
            res_df[["cv_recall", "Accuracy", "Precision", "Recall", "F1", "AUC-ROC"]],
            use_container_width=True,
        )

        st.subheader("Prédiction")
        c1, c2, c3, c4 = st.columns(4)
        ph = c1.number_input("PH", 0.0, 14.0, 6.5)
        temp = c2.number_input("Température (°C)", 0.0, 50.0, 27.0)
        do = c3.number_input("Oxygène dissous (mg/L)", 0.0, 30.0, 7.0)
        turbidity = c4.number_input("Turbidité (NTU)", 0.0, 100.0, 30.0)
        model_choice = st.selectbox(
            "Modèle à utiliser", list(st.session_state["m2_models"].keys()), index=2, key="m2_model"
        )

        if st.button("Prédire la qualité"):
            pred, proba = module2.predict_quality(
                st.session_state["m2_models"], st.session_state["m2_scaler"], model_choice, ph, temp, do, turbidity
            )
            label = "✅ Bonne qualité" if pred == 1 else "⚠️ Mauvaise qualité"
            st.metric(label, f"Probabilité bonne qualité : {proba:.1%}")
    else:
        st.info("Cliquez sur le bouton ci-dessus pour entraîner les modèles.")

# =====================================================================================
# MODULE 3 — STL + LSTM : anticipation à H+1
# =====================================================================================
with tab_m3:
    st.header("Module 3 — Anticipation de la qualité de l'eau à H+1")
    st.caption(
        "Décomposition STL (tendance / saisonnalité) puis LSTM sur la composante de tendance, "
        "un modèle par variable (section 5.3)."
    )
    st.info(
        "🔒 La composante saisonnière $S_t$ est calculée **une seule fois** par la décomposition "
        "STL, puis stockée telle quelle dans `seasonality.db` (jour de l'année 1-366 × heure "
        "0-23). Cette table est **immuable** : elle n'est ni modifiable, ni affichée dans "
        "l'interface — elle sert uniquement en interne à reconstituer la prédiction finale "
        "(tendance LSTM + $S_t$)."
    )

    if st.button("🚀 Décomposer (STL) et entraîner les LSTM du Module 3"):
        with st.spinner("Génération de la série temporelle horaire et décomposition STL..."):
            trend_df = module3.decompose_and_store()
        with st.spinner("Entraînement des 4 modèles LSTM (peut prendre 1-2 minutes)..."):
            results, models, scalers = module3.train_lstm_models(trend_df)
        st.session_state["m3_trend_df"] = trend_df
        st.session_state["m3_results"] = results
        st.session_state["m3_models"] = models
        st.session_state["m3_scalers"] = scalers
        st.success("Décomposition STL effectuée, saisonnalité stockée en base, modèles LSTM entraînés.")

    if "m3_results" in st.session_state:
        res_df = pd.DataFrame(st.session_state["m3_results"]).T
        st.dataframe(res_df[["RMSE", "MAE", "MAPE", "R2"]], use_container_width=True)

        st.subheader("Prédiction à H+1")
        trend_df = st.session_state["m3_trend_df"]
        c1, c2, c3 = st.columns(3)
        var = c1.selectbox("Variable", module3.VARS)
        date_choice = c2.number_input("Jour de l'année (1-366)", min_value=1, max_value=365, value=100)
        heure_choice = c3.number_input("Heure (0-23)", min_value=0, max_value=23, value=12)

        if st.button("Prédire H+1"):
            out = module3.predict_h_plus_1(
                st.session_state["m3_models"],
                st.session_state["m3_scalers"],
                trend_df,
                var,
                int(date_choice),
                int(heure_choice),
            )
            if out is None:
                st.warning(
                    "Pas assez d'historique disponible pour ce point (il faut au moins 3 pas de "
                    "temps précédents dans la série simulée)."
                )
            else:
                final_value, pred_trend, s_val, next_date, next_hour = out
                st.metric(
                    f"{var} prévu — jour {next_date}, {next_hour}h",
                    f"{final_value:.2f}",
                )
                st.caption(f"= tendance LSTM ({pred_trend:.2f}) + saisonnalité S_t stockée en base ({s_val:.2f})")
    else:
        st.info("Cliquez sur le bouton ci-dessus pour lancer la décomposition STL et l'entraînement des LSTM.")
