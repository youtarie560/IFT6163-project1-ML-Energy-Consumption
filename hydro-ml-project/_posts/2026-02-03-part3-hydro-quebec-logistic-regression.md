---
layout: post
title: "Partie 3: Ingénierie des Caractéristiques (Feature Engineering)"
date: 2026-02-03 12:00:00 -0500
categories: machine-learning project
use_math: true
---

Un modèle linéaire standard suppose que chaque observation est indépendante. Or, dans une série temporelle (comme ce problème), 
la meilleure prédiction pour "maintenant" est souvent "ce qu'il s'est passé il y a une heure".
En d'autre terme, jusqu'à présent, notre modèle oublie ou ne retient pas beaucoup de détails. C'est-à-dire que pour prédire la consommation 
à 8h00, il ne regardait que la météo de 8h00. Il ignorait si le chauffage tournait à temps plein depuis 3 heures ou s'il venait de s'allumer.

Pour corriger cela, j'ai créé 3 nouvelles observations:

### 1. La Mémoire (Lags)
* **Lag 1h :** La consommation de l'heure précédente.
* **Lag 24h :** La consommation à la même heure la veille. Cela capture les habitudes de vie (on consomme plus le matin et le soir).

### 2. La Tendance (Moyennes Mobiles)
En calculant la moyenne sur 6 heures, nous lissons le bruit pour donner au modèle une idée de la consommation actuelle.

### 3. La Physique (Degrés-Jours)
La relation entre température et chauffage n'est pas linéaire :
* À 25°C ou 20°C, le chauffage est éteint (consommation constante).
* En dessous de 18°C, chaque degré perdu augmente la consommation.
* Un modèle linéaire simple comme: $y = ax + b$ n'est pas approprie pour ce probleme. Si vous lui donnez juste la température brute, 
il va penser : "Plus il fait chaud, moins on consomme". Du coup, quand il fait 30°C l'été, le modèle linéaire va prédire une consommation négative (ou absurdement basse), 

Donc, modéliser ce comportement introduit de la non-linéarité sans avoir besoin d'un polynôme complèxe telle que **HDD (Heating Degree Days)**: $\max(18 - T, 0)$ aide le modèle
 à comprendre ce "coude" à 18°C. Puisque le modèle linéaire ne peut pas "plier" sa droite pour 
suivre le coude, nous devons transformer la donnée pour lui. Nous utilisons la formule :

| Scénario | Température ($T$) | Calcul ($18-T$) | Résultat $\max(18-T, 0)$ | Interprétation pour le modèle           |
| :--- | :--- | :--- |:-------------------------|:----------------------------------------|
| **Été** | 30°C | $18 - 30 = -12$ | **0**                    | Pas besoin de chauffer                  |
| **Confort** | 19°C | $18 - 19 = -1$ | **0**                    | Toujours pas besoin de chauffer         |
| **Automne** | 10°C | $18 - 10 = 8$ | **8**                    | On a besoin de "8 unités" de chauffage  |
| **Hiver** | -20°C | $18 - (-20) = 38$ | **38**                   | On a besoin de "38 unités" de chauffage |

Grâce à cette transformation, la relation devient purement proportionnelle et positive
- Si HDD = 0, Consommation de chauffage = 0.
- Si HDD augmente, Consommation augmente.

Cette partie s'agit de transformer un problème non-linéaire en un problème linéaire.
- Sans HDD : Le modèle essaie de faire passer une droite au milieu d'une courbe, ce qui crée des erreurs partout.
- Avec HDD : La relation devient positive et proportionnelle (Si HDD augmente, Conso augmente ; si HDD=0, Consommation=0).
>> Conseil pour la suite. Une fois ces caractéristiques créées, on va ré-entraîner le modèle (Partie 4 ou plus tard).
>> on va voir que le score $R^2$ va augmenter (probablement au-dessus de 0.90) grâce au lag1.
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