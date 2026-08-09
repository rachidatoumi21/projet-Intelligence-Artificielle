# IFT3335 — Classification d’images avec un MLP

Projet réalisé dans le cadre du cours **IFT3335**.  
L’objectif est d’étudier la **classification d’images** avec un réseau de neurones multicouche (**MLP — Multi-Layer Perceptron**) et d’analyser l’effet de plusieurs hyperparamètres sur l’apprentissage et la généralisation.

Le projet utilise le jeu de données **Fashion-MNIST**, composé d’images en niveaux de gris de taille **28 × 28 pixels** réparties en **10 classes** de vêtements.

## Objectifs du projet

Le travail est divisé en deux parties principales :

### Partie 1 — Composantes d’un réseau de neurones

Implémentation de plusieurs éléments fondamentaux de l’apprentissage automatique :

- neurone artificiel ;
- couche entièrement connectée ;
- fonctions d’activation :
  - Sigmoid ;
  - Tanh ;
  - Leaky ReLU ;
  - Softmax ;
- fonctions de perte :
  - MAE ;
  - MSE ;
  - Log-Cosh ;
  - Cross-Entropy ;
- réseau de neurones multicouche (**MLP**) ;
- boucle d’entraînement et évaluation du modèle.

### Partie 2 — Expérimentation

Plusieurs modèles sont entraînés afin d’étudier l’influence des hyperparamètres suivants :

1. normalisation des données ;
2. nombre de couches cachées ;
3. taux d’apprentissage ;
4. fonction de perte ;
5. taille des couches cachées ;
6. taille des minibatchs.

Pour chaque expérience, les métriques d’entraînement et de test sont sauvegardées puis utilisées pour tracer les courbes de **précision** et de **perte**.

## Jeu de données

Le projet utilise **Fashion-MNIST**.

Chaque image :

- est en niveaux de gris ;
- possède une résolution de `28 × 28` pixels ;
- appartient à l’une des 10 catégories suivantes :

```text
T-shirt/top
Trouser
Pullover
Dress
Coat
Sandal
Shirt
Sneaker
Bag
Ankle boot
```

## Architecture de référence

Le modèle Baseline utilise :

```text
Entrée : 784 neurones
Couche cachée 1 : 64 neurones
Couche cachée 2 : 64 neurones
Sortie : 10 neurones
```

Configuration principale :

```text
Normalisation : Oui
Learning rate : 0.05
Loss : Cross-Entropy
Batch size : 64
Epochs : 20
```

## Expériences

| Expérience | Paramètre étudié | Configuration |
|---|---|---|
| Baseline | Modèle de référence | 64-64, lr=0.05, CE, batch=64 |
| 1 | Normalisation | données non normalisées |
| 2 | Nombre de couches | 1 couche cachée de 70 neurones |
| 3 | Learning rate | lr=0.1 |
| 4 | Learning rate | lr=0.01 |
| 5 | Fonction de perte | Hinge Loss |
| 6 | Fonction de perte | MSE |
| 7 | Taille des couches | 48 → 300 |
| 8 | Taille des couches | 300 → 48 |
| 9 | Taille du minibatch | batch size = 2 |

## Résultats principaux

Les expériences montrent notamment que :

- la **normalisation** améliore fortement la stabilité et les performances du modèle ;
- le Baseline normalisé atteint environ **87.17 %** de précision sur le jeu de test ;
- avec un **learning rate de 0.1**, le modèle atteint environ **87.49 %**, soit le meilleur résultat observé parmi les taux testés ;
- une seule couche cachée de 70 neurones donne des performances très proches du Baseline à deux couches ;
- **Cross-Entropy** et **Hinge Loss** donnent des précisions similaires ;
- **MSE** est nettement moins adaptée à cette tâche de classification multiclasse ;
- les architectures `48 → 300` et `300 → 48` donnent des performances proches du Baseline ;
- un minibatch très petit apprend rapidement au début, mais produit un entraînement plus bruité et moins stable.

## Visualisations

Le notebook génère des graphiques permettant de comparer les modèles selon :

- la précision d’entraînement ;
- la précision de test ;
- la perte d’entraînement ;
- la perte de test.

Il permet également de visualiser :

- des exemples aléatoires du jeu de données ;
- des images mal classifiées par le modèle ;
- la classe réelle et la classe prédite.

## Structure du projet

```text
.
├── Part1.py
├── MLP.py
├── experiments.ipynb
├── Partie2.pdf
├── results.pkl
├── data/
└── README.md
```

### `Part1.py`

Contient les principales composantes de base du réseau de neurones ainsi que l’algorithme d’entraînement.

### `MLP.py`

Contient l’architecture du réseau de neurones multicouche utilisée pour les expériences.

### `experiments.ipynb`

Notebook utilisé pour :

- charger Fashion-MNIST ;
- configurer les expériences ;
- entraîner les modèles ;
- sauvegarder les métriques ;
- comparer les résultats ;
- générer les graphiques.

### `Partie2.pdf`

Rapport présentant l’analyse des différentes expériences.

## Technologies utilisées

- Python
- PyTorch
- Torchvision
- NumPy
- Matplotlib
- Jupyter Notebook / VS Code

## Installation

Cloner le dépôt :

```bash
git clone https://github.com/VOTRE-UTILISATEUR/VOTRE-DEPOT.git
cd VOTRE-DEPOT
```

Installer les dépendances :

```bash
pip install torch torchvision numpy matplotlib jupyter
```

## Exécution

Ouvrir le notebook :

```bash
jupyter notebook experiments.ipynb
```

ou l’exécuter directement dans **Visual Studio Code** avec l’extension Jupyter.

Le jeu de données Fashion-MNIST est téléchargé automatiquement par `torchvision` lors de la première exécution.

## Auteur

**Rachida Toumi**

Projet académique — IFT3335
