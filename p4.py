import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, Ridge, RidgeCV, LogisticRegression
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
import warnings
from p3 import creer_caracteristiques

train = pd.read_csv('energy_train.csv')
test = pd.read_csv('energy_test.csv')
# Appliquer aux données
train_eng = creer_caracteristiques(train)
test_eng = creer_caracteristiques(test)

# Supprimer les lignes avec NaN (dues aux retards)
train_eng = train_eng.dropna()
test_eng = test_eng.dropna()

# Définissez vos caractéristiques pour la régression
# MODIFIEZ CETTE LISTE selon vos caractéristiques créées en Partie 3
# IMPORTANT: clients_connectes est une variable très importante!
features_reg = [
    'temperature_ext', 'humidite', 'vitesse_vent', 'irradiance_solaire',
    'heure_sin', 'heure_cos', 'mois_sin', 'mois_cos',
    'jour_semaine_sin', 'jour_semaine_cos',
    'est_weekend', 'est_ferie',
    'clients_connectes',  # Ne pas oublier!
    'energie_lag1',
    'energie_lag24',
    'energie_rolling_mean_6h',
    'degre_jour_chauffage',
    'interaction_temp_vent'
]

# Vérifier que toutes les colonnes existent
features_disponibles = [f for f in features_reg if f in train_eng.columns]
print(f"Caractéristiques utilisées: {len(features_disponibles)}")

X_train_raw = train_eng[features_disponibles].values
y_train_reg = train_eng['energie_kwh'].values
X_test_raw = test_eng[features_disponibles].values
y_test_reg = test_eng['energie_kwh'].values


# scaling avant
scaler = StandardScaler()
X_train_reg = scaler.fit_transform(X_train_raw)
X_test_reg = scaler.transform(X_test_raw)

# Modèle OLS (baseline)
print("\n--- Entraînement OLS ---")
model_ols = LinearRegression()
model_ols.fit(X_train_reg, y_train_reg)
y_pred_ols = model_ols.predict(X_test_reg)

print("OLS (baseline):")
print(f"OLS  R² train: {model_ols.score(X_train_reg, y_train_reg):.4f}")
print(f"OLS  R² test:  {r2_score(y_test_reg, y_pred_ols):.4f}")
print(f"OLS  RMSE test: {np.sqrt(mean_squared_error(y_test_reg, y_pred_ols)):.4f}")

print("\n--- Entraînement Ridge (avec TimeSeriesSplit) ---")
# Modèle Ridge avec validation croisée
# ATTENTION: Utilisez TimeSeriesSplit pour les données temporelles!
from sklearn.model_selection import TimeSeriesSplit

alphas = [0.01, 0.1, 1, 10, 100, 1000]
tscv = TimeSeriesSplit(n_splits=5)

model_ridge = RidgeCV(alphas=alphas, cv=tscv)
model_ridge.fit(X_train_reg, y_train_reg)
y_pred_ridge = model_ridge.predict(X_test_reg)

print(f"\nRidge (λ={model_ridge.alpha_}):")
print(f"Ridge  R² train: {model_ridge.score(X_train_reg, y_train_reg):.4f}")
print(f"Ridge  R² test:  {r2_score(y_test_reg, y_pred_ridge):.4f}")
print(f"Ridge  RMSE test: {np.sqrt(mean_squared_error(y_test_reg, y_pred_ridge)):.4f}")

# Comparaison des coefficients OLS vs Ridge
coef_comparison = pd.DataFrame({
    'Caractéristique': features_disponibles,
    'OLS': model_ols.coef_,
    'Ridge': model_ridge.coef_
})
coef_comparison['Réduction (%)'] = 100 * (1 - np.abs(coef_comparison['Ridge']) / (np.abs(coef_comparison['OLS']) + 1e-8))
coef_comparison = coef_comparison.sort_values('Réduction (%)', ascending=False)

print("\nComparaison des coefficients (triés par réduction):")
print(coef_comparison.to_string(index=False))
print("\n--- Top 5 des coefficients les plus réduits par Ridge ---")
print(coef_comparison.head(5).to_string(index=False))

print("\n--- Top 5 des coefficients les plus importants (Ridge) ---")
# On trie par valeur absolue pour voir l'importance réelle
coef_comparison['Abs_Ridge'] = np.abs(coef_comparison['Ridge'])
print(coef_comparison.sort_values('Abs_Ridge', ascending=False)[['Caractéristique', 'Ridge']].head(5).to_string(index=False))