---
layout: post
title:  "Partie 1: Implémentation OLS"
date:   2026-02-03 08:00:00 -0500
categories: machine-learning project
use_math: true
---

## 1. Partie 1
Moindres carrés ordinaires (MCO) ou **Ordinary Least Squares (OLS)** en anglais pour predire la consommation énergitique de
Hydro-Québec. <br>
En théorie, nous pouvons modéliser ceci sous-forme de notation matricielle pour _**N**_ observations et _**d**_ charactéristiques (features) où le biais _**b**_
peut être absorbé en ajoutant une colonne de 1 à $$X$$
.

$$
X = \begin{bmatrix}
x_{1,1} & x_{1,2} & \cdots & x_{1,d} & 1 \\
x_{2,1} & x_{2,2} & \cdots & x_{2,d} & 1 \\
\vdots & \vdots & \vdots & \ddots & \vdots \\
x_{N,1} & x_{N,2} & \cdots & x_{N,d} & 1
\end{bmatrix}
$$
,
$$
y = \begin{bmatrix}
y_{1} \\
y_{2} \\
\vdots \\
y_{N}
\end{bmatrix}
$$
<br>
Si $$X^T X$$ est inversible, la solution est $$\hat{\beta} = (X^T X)^{-1} X^T y$$. 
Ceci est une partie du code python: <br>

```python
def ols_fit(X, y):
    ones_col = np.ones((X.shape[0], 1))
    X_aug = np.hstack([ones_col, X]) # size (n, p+1), concatener la columne de "1" pour le biais
    # Terme de gauche : X^T * X
    terme_gauche = X_aug.T @ X_aug

    # Terme de droite : X^T * y
    terme_droite = X_aug.T @ y
    beta = np.linalg.solve(terme_gauche, terme_droite)
```
Il est conceillé explicitement d'utiliser ```np.linalg.solve``` au lieu de ```np.linalg.inv```.
- Plus rapide
- Plus précis (moins d'erreurs d'arrondi)

Finalement, les prédictions pour tous les exemples s'écrivent: $$\hat{y} = X{\beta}$$.
C'est ce que la fonction ```ols_predict``` fait:
```python
def ols_predict(X, beta):
    return X @ beta
```
<br>
Bien que cette implémentation MCO correspond parfaitement à Scikit-Learn,
```
Comparaison OLS implémenté vs sklearn:
  Intercept - Vous: 234.8557, sklearn: 234.8557
  Coefficients proches: True

R² sur test: -2.2572
R² sur test (de sklearn): -2.2572
```
le modèle a donné un $$R^2$$ négatif sur l'ensemble de test. Cela confirme le changement de distribution assez important.
C'est-à-dire que le modèle à été entraîné sur les données hivernales qui ce dernier apprend une base de référence élevée
 (moyenne de consommation $\approx$ 235 kWh) qui surestime la consommation dans l'ensemble de test printemps/été (moyenne de consommation $\approx$ 84 kWh).


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