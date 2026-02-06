---
layout: post
title: "Partie 4: Régression Ridge"
date: 2026-02-03 12:00:00 -0500
categories: machine-learning project
use_math: true
---

Avec l'ajout de nos nouvelles caractéristiques (Partie 3), il y a un problème : la **multicolinéarité**.
* `temperature_ext` est fortement corrélée avec `degre_jour_chauffage`.
* `energie_lag1` est corrélée avec `energie_rolling_mean`.


OLS gère mal ce dernier car il a tendance à donner des coefficients énormes et opposés pour compenser, ce qui rend le modèle instable.

### Pourquoi Ridge ?
La régression Ridge ajoute une pénalité ($L_2$) à la fonction de coût :
$$J(\theta) = \text{MSE} + \lambda \sum \theta_i^2$$

Cela force le modèle à "rétrécir" les coefficients inutiles vers zéro et à répartir le poids entre les variables corrélées.
En appliquant ```StandardScaler```, nous transformons toutes les variables pour qu'elles aient une moyenne de 
0 et un écart-type de 1 ($$ z = \frac{x - \mu}{\sigma} $$). La pénalité 
Ridge s'applique sur l'importance réelle de la variable, et non sur son unité de mesure.
```python
# scaling avant
scaler = StandardScaler()
X_train_reg = scaler.fit_transform(X_train_raw)
X_test_reg = scaler.transform(X_test_raw)
```
C'est-à-dire que pour qu'une petite variable comme heure_sin (0 à 1) ait un impact sur la consommation (qui est en milliers de kWh), 
elle a besoin d'un coefficient énorme. La régression Ridge cherche à minimiser la somme des coefficients au carré $$\lambda \sum \theta^2$$.

### Validation Croisée Temporelle (TimeSeriesSplit)
Pour trouver le bon paramètre de pénalité $\lambda$, nous ne pouvons pas utiliser une validation croisée aléatoire (k-fold), 
car cela briserait la chronologie. Nous utilisons **TimeSeriesSplit** où on entraîne sur [Jan-Mars] et on valide sur [Avril], puis on entraîne sur [Jan-Avril] et on valide sur [Mai], etc.

### Résultats

Les performances sur l'ensemble de test montrent une amélioration claire :

| Modèle | $R^2$ (Test data) | RMSE (Test data) | Interprétation                                         |
| :--- |:------------------|:-----------------|:-------------------------------------------------------|
| **OLS (Baseline)** | 0.6930            | 38.63            | Tendance à sur-ajuster le bruit (coefficients énormes) |
| **Ridge ($\lambda=100$)** | **0.7158**        | **37.17**        | **+2.3% de performance**, modèle généralise mieux      |

*Tableau 1 : Comparaison de la performance (OLS vs Ridge)* <br>
#### Ridge "punit" la redondance
Ridge a détecté que la température et le chauffage racontaient la même histoire. Il a réduit l'importance de la température brute (bruitée) au profit du degré-jour (plus physique).

| Caractéristique | OLS | Ridge | Réduction (%) |
| :--- | :--- | :--- | :--- |
| `energie_lag24` | -1.85 | -0.61 | **-67.0%** |
| `temperature_ext` | 56.73 | 19.99 | **-64.7%** |
| `degre_jour_chauffage` | 79.30 | 45.54 | **-42.6%** |
| `interaction_temp_vent` | -8.29 | -5.75 | -30.6% |


*Tableau 2: Top 5 des réductions de coefficients* <br>
En regardant les coefficients, vous remarquerez que certaines valeurs sont négatives. Cela ne signifie pas que ces
caractéristiques sont inutiles. Au contraire, cela indique une **relation inverse**.

Prenons l'exemple concret de la ligne `interaction_temp_vent` du tableau, qui a un coefficient négatif (**-5.75**).

Cette variable est le résultat de : $Température \times VitesseDuVent$.
Ce coefficient négatif permet au modèle de comprendre l'effet physique du vent selon la saison :

1.  **En Hiver (Chauffage) :**
    * Imaginons qu'il fait **-10°C** avec un vent de **20 km/h**.
    * L'interaction vaut : $-10 \times 20 = -200$.
    * Le modèle calcule : $(-5.75) \times (-200) = \mathbf{+1150} \text{ kWh}$.
    * **Résultat :** Le signe "moins" du coefficient annule le "moins" de la température. Le modèle comprend que le vent en hiver **augmente** la consommation (le facteur éolien refroidit la maison).

2.  **En Été (Climatisation) :**
    * Imaginons qu'il fait **+30°C** avec le même vent.
    * L'interaction vaut : $30 \times 20 = +600$.
    * Le modèle calcule : $(-5.75) \times (600) = \mathbf{-3450} \text{ kWh}$.
    * **Résultat :** Le modèle réduit la prévision. Le vent aide à refroidir les bâtiments naturellement, réduisant le besoin de climatisation.

> **Analyse :** OLS donnait un poids énorme à la température (56.7) ET au chauffage (79.3). C'est le symptôme classique de la multicolinéarité. Ridge a calmé le jeu en réduisant la température à 20.0.

#### 2. Les vrais moteurs de la consommation (MVP)
Quelles sont les variables les plus importantes pour le modèle final ? Ce n'est pas la météo, mais l'inertie du système.

*Top 5 des coefficients les plus importants (Valeur Absolue) :*

| Caractéristique | Coefficient Ridge | Rôle |
| :--- | :--- | :--- |
| **1. `energie_rolling_mean_6h`** | **150.83** | **Tendance (Inertie)** |
| 2. `degre_jour_chauffage` | 45.54 | Physique (Froid) |
| 3. `clients_connectes` | 34.60 | Taille du réseau |
| 4. `heure_cos` | -33.21 | Cycle journalier |
| 5. `energie_lag1` | -28.15 | Mémoire immédiate |


> **Le MVP :** Avec un coefficient de **150.8**, la moyenne mobile sur 6h écrase tout le reste. Pour prédire la consommation future, savoir "combien on consommait il y a 2h" est plus utile que de savoir "combien il fait dehors".

## Conclusion
Ce projet surmonte les défis de la prévision énergétique :

1.  Un modèle simple (Partie 1) échoue car il ne comprend pas les changements de saison.
2.  L'ajout du **HDD (Partie 3)** a permis de linéariser la relation Température/Consommation.
3.  Les variables autorégressives (moyennes mobiles) sont les prédicteurs les plus puissants.
4.  Face à des variables corrélées, Ridge a amélioré la généralisation de **2.3%**

Pour aller plus loin, un modèle non-linéaire comme **Random Forest** ou **XGBoost** pourrait capturer les interactions
complèxes que Ridge ne peut pas voir.














<script>
  MathJax = {
    tex: {
      inlineMath: [['$', '$'], ['\\(', '\\)']],
      displayMath: [['$$', '$$'], ['\\[', '\\]']],
      processEscapes: true
    }
  };
</script>
<script id="MathJax-script" async src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"></script>