"""
Génération de données synthétiques cohérentes avec les distributions décrites
dans le mémoire (section 4.3), en l'absence des jeux de données réels :
- Module 1 : dataset synthétique par nature dans le mémoire lui-même.
- Modules 2/3 : dataset Kaggle non fourni ici -> simulé avec des distributions
  équivalentes à celles présentées (violin plots, histogrammes, corrélations).
"""
import numpy as np
import pandas as pd

RNG_SEED = 42

TEXTURE_ENC = {"Sableux": 0, "Limoneux": 1, "Argileux": 2}
COULEUR_ENC = {"Orange_Rouge": 1, "Brun_Grisatre": 2, "Noir": 3}
IRRIGATION_ENC = {"Pluvial": 0, "Continu": 1}
ENGRAIS_ENC = {"Nul": 0, "Moyen": 1, "Fort": 2}
TEMP_ENC = {"TresFroid": 0, "Froid": 1, "Humide": 2, "Chaud": 3, "TresChaud": 4}


def generate_module1_data(n=2000, seed=RNG_SEED):
    rng = np.random.default_rng(seed)

    texture_sol = rng.choice(list(TEXTURE_ENC), size=n, p=[0.35, 0.35, 0.30])
    couleur_sol = rng.choice(list(COULEUR_ENC), size=n, p=[0.35, 0.35, 0.30])
    type_irrigation = rng.choice(list(IRRIGATION_ENC), size=n, p=[0.4, 0.6])
    niveau_engrais = rng.choice(list(ENGRAIS_ENC), size=n, p=[0.3, 0.4, 0.3])
    temperature = rng.choice(list(TEMP_ENC), size=n, p=[0.15, 0.2, 0.2, 0.3, 0.15])

    hauteur_bin = rng.choice(["5-8", "9-14", "15-20"], size=n, p=[0.3, 0.4, 0.3])
    hauteur_eau = np.array(
        [
            rng.integers(5, 9) if b == "5-8" else rng.integers(9, 15) if b == "9-14" else rng.integers(15, 21)
            for b in hauteur_bin
        ]
    )

    t_enc = np.vectorize(TEXTURE_ENC.get)(texture_sol)
    c_enc = np.vectorize(COULEUR_ENC.get)(couleur_sol)
    i_enc = np.vectorize(IRRIGATION_ENC.get)(type_irrigation)
    e_enc = np.vectorize(ENGRAIS_ENC.get)(niveau_engrais)
    tp_enc = np.vectorize(TEMP_ENC.get)(temperature)

    density = (
        10
        + t_enc * 6.0
        + c_enc * 2.5
        + i_enc * 8.0
        + (hauteur_eau - 5) * 0.9
        + e_enc * 6.5
        + np.where(tp_enc == 3, 8, 0)
        - np.where(tp_enc == 0, 6, 0)
        + rng.normal(0, 4, size=n)
    )
    density = np.clip(density, 8, 55).round().astype(int)

    return pd.DataFrame(
        {
            "Texture_sol": texture_sol,
            "Couleur_sol": couleur_sol,
            "Type_irrigation": type_irrigation,
            "Hauteur_Eau": hauteur_eau,
            "Niveau_Engrais": niveau_engrais,
            "Temperature_Actuelle": temperature,
            "Densite_Tilapia_Are": density,
        }
    )


def generate_water_quality_dataset(n=8000, seed=RNG_SEED):
    """Simule (PH, TEMP, DO, TURBIDITY, label), déséquilibre ~73%/27% bonne/
    mauvaise qualité, cohérent avec les figures 10-15 du mémoire."""
    rng = np.random.default_rng(seed)
    n_good = int(n * 0.73)
    n_bad = n - n_good

    good = pd.DataFrame(
        {
            "PH": rng.normal(6.2, 0.6, n_good).clip(4.5, 9),
            "TEMP": rng.normal(26, 5, n_good).clip(10, 40),
            "DO": rng.normal(7, 1.6, n_good).clip(2, 15),
            "TURBIDITY": rng.normal(28, 8, n_good).clip(10, 60),
            "label": 1,
        }
    )
    bad = pd.DataFrame(
        {
            "PH": rng.normal(7.9, 0.7, n_bad).clip(4.5, 9),
            "TEMP": rng.normal(35, 5, n_bad).clip(10, 45),
            "DO": np.concatenate(
                [
                    rng.normal(1.5, 1, n_bad // 2).clip(0, 4),
                    rng.normal(21, 3, n_bad - n_bad // 2).clip(18, 26),
                ]
            ),
            "TURBIDITY": rng.normal(55, 10, n_bad).clip(30, 85),
            "label": 0,
        }
    )
    df = pd.concat([good, bad], ignore_index=True).sample(frac=1, random_state=seed).reset_index(drop=True)
    return df


def generate_hourly_series(n_days=365, seed=RNG_SEED):
    """Série temporelle horaire sur 1 an (PH, TEMP, DO, TURBIDITY), avec cycle
    journalier + dérive saisonnière lente, pour alimenter STL + LSTM (Module 3)."""
    rng = np.random.default_rng(seed)
    n = n_days * 24
    t = np.arange(n)
    hour = t % 24
    day = t // 24 + 1

    def make_series(mean, daily_amp, annual_amp, noise_std, annual_phase=0.0):
        daily = daily_amp * np.sin(2 * np.pi * hour / 24)
        annual = annual_amp * np.sin(2 * np.pi * day / 365 + annual_phase)
        noise = rng.normal(0, noise_std, n)
        return mean + daily + annual + noise

    temp = make_series(26, 4, 3, 0.8)
    ph = make_series(6.3, 0.3, 0.2, 0.15)
    do = make_series(7, 1.2, 0.8, 0.4, annual_phase=np.pi)
    turbidity = make_series(30, 5, 6, 2.0)

    return pd.DataFrame(
        {
            "date": day,
            "heure": hour,
            "PH": ph,
            "TEMP": temp,
            "DO": do,
            "TURBIDITY": turbidity,
        }
    )
