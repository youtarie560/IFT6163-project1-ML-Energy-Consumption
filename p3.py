import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, Ridge, RidgeCV, LogisticRegression
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
import warnings


def creer_caracteristiques(df: pd.DataFrame):
    """
    Crée des caractéristiques supplémentaires.

    VOUS DEVEZ IMPLÉMENTER AU MOINS 3 NOUVELLES CARACTÉRISTIQUES.

    Idées:
    - Retards: df['energie_kwh'].shift(1), shift(24)
    - Moyennes mobiles: df['energie_kwh'].rolling(6).mean()
    - Interactions: df['temperature_ext'] * df['heure_cos']
    - Degré-jours de chauffage: np.maximum(18 - df['temperature_ext'], 0)
    """
    df = df.copy()
    # 1. Retards (Lags) - La mémoire du système
    # Ce qu'on consommait il y a 1 heure (Inertie thermique très forte)
    df['energie_lag1'] = df['energie_kwh'].shift(1)

    # Ce qu'on consommait il y a 24 heures (Cycle d'activité humaine)
    # Si c'est 8h du matin, on regarde la consommation de 8h hier.
    df['energie_lag24'] = df['energie_kwh'].shift(24)

    # Moyenne des 6 dernières heures pour lisser le bruit
    df['energie_rolling_mean_6h'] = df['energie_kwh'].rolling(window=6).mean()

    # 3. Interactions Physiques - La réalité du chauffage
    # Degrés-Jours de Chauffage (HDD) : On ne chauffe que si T < 18°C.
    # C'est une transformation non-linéaire cruciale : la relation Conso vs Temp est plate au-dessus de 18°C.
    df['degre_jour_chauffage'] = np.maximum(18 - df['temperature_ext'], 0)

    # Interaction Température x Vent (Refroidissement éolien)
    # Le vent augmente la perte de chaleur, surtout quand il fait froid.
    df['interaction_temp_vent'] = df['temperature_ext'] * df['vitesse_vent']

    return df

train = pd.read_csv('energy_train.csv')
test = pd.read_csv('energy_test.csv')
# Appliquer aux données
train_eng = creer_caracteristiques(train)
test_eng = creer_caracteristiques(test)

# Supprimer les lignes avec NaN (dues aux retards)
train_eng = train_eng.dropna()
test_eng = test_eng.dropna()

print(f"Nouvelles colonnes: {[c for c in train_eng.columns if c not in train.columns]}")