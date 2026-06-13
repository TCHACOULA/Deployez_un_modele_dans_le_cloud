# Projet 8 — Déployez un modèle dans le cloud

> **Parcours Data Scientist — OpenClassrooms**  
> Auteur : Saliou TCHACOULA  
> Soutenance : Septembre 2024

---

## Contexte

Ce projet est réalisé dans le cadre d'une mission de Data Scientist au sein de la start-up **"Fruits!"**, une jeune entreprise de l'**AgriTech**.

L'objectif de la start-up est de développer une **application mobile** permettant aux utilisateurs de :
- Prendre en photo un fruit et obtenir automatiquement des informations sur ce fruit.
- Sensibiliser à la biodiversité des fruits.
- Disposer d'un premier moteur de classification d'images de fruits.

La mission assignée est de **construire une architecture Big Data dans le cloud** capable de traiter des images de fruits à grande échelle, en s'appuyant sur les services AWS et Apache Spark.

---

## Architecture Cloud (AWS)

```
┌──────────────────────────────────────────────────────────────────┐
│                         AWS Cloud                                 │
│                                                                    │
│  ┌─────────────┐     ┌──────────────────────────────────────┐    │
│  │  Amazon S3  │◄────│           Amazon EMR                  │    │
│  │             │     │  (Elastic MapReduce)                   │    │
│  │ /Test       │────►│                                        │    │
│  │ /Results    │     │  1 instance primaire m5.xlarge         │    │
│  │ /PCAResults │◄────│  2 instances principales m5.xlarge     │    │
│  └─────────────┘     │                                        │    │
│                       │  Logiciels : Spark, Hadoop,           │    │
│  ┌─────────────┐     │  JupyterHub, TensorFlow               │    │
│  │  Amazon EC2 │     └──────────────────────────────────────┘    │
│  │  (accès SSH)│                                                   │
│  └─────────────┘                                                   │
└──────────────────────────────────────────────────────────────────┘
```

### Services AWS utilisés

**Amazon S3 (Simple Storage Service)**
Stockage de toutes les données du projet :
- `/Test` : 22 819 images de fruits (`.jpg`)
- `/Results` : vecteurs de features extraits (format Parquet)
- `/PCAResults` : features après réduction de dimension PCA (format Parquet)

**Amazon EMR (Elastic MapReduce)**
Cluster Spark géré par AWS :
- 1 instance primaire `m5.xlarge` + 2 instances principales `m5.xlarge`
- Logiciels installés : Spark, Hadoop, JupyterHub, TensorFlow
- Packages Python installés via une **action d'amorçage (bootstrap)** au démarrage du cluster : NumPy, Pandas, Pillow, Keras...
- Persistance des notebooks activée : sauvegarde automatique dans S3

**Amazon EC2 (Elastic Compute Cloud)**
Instances virtuelles sous-jacentes au cluster EMR, configurées avec une paire de clés SSH pour l'accès distant sécurisé.

---

## Pipeline de traitement

Le notebook couvre deux environnements d'exécution :

| Phase | Environnement | Images traitées |
|-------|--------------|-----------------|
| Partie 1 | Local (simulation Spark multi-cœurs) | 330 images (extrait `Test1`) |
| Partie 2 | Cluster AWS EMR (production) | 22 819 images (dossier `Test`) |

---

## Description du notebook

### Partie 1 — Exécution locale (simulation du cluster)

#### Étape 1.1 — Import des librairies

```python
import pandas as pd
from PIL import Image
import numpy as np
import io, os

import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras import Model

from pyspark.sql.functions import col, pandas_udf, PandasUDFType, element_at, split
from pyspark.sql import SparkSession
```

#### Étape 1.2 — Définition des chemins

```python
PATH      = os.getcwd()
PATH_Data = PATH + '/data/Test'
PATH_Result = PATH + '/data/Results'
```

Les données (330 images) sont stockées localement dans `./data/Test`.
Les résultats seront écrits dans `./data/Results` au format Parquet.

#### Étape 1.3 — Création de la SparkSession

```python
spark = (SparkSession
             .builder
             .appName('P8')
             .master('local')
             .config("spark.sql.parquet.writeLegacyFormat", 'true')
             .getOrCreate())
sc = spark.sparkContext
```

- `.master('local')` : utilise tous les cœurs CPU disponibles sur la machine.
- `.config("spark.sql.parquet.writeLegacyFormat", 'true')` : assure la compatibilité Parquet entre Spark et PyArrow/Pandas.
- `sc = spark.sparkContext` : le SparkContext est nécessaire pour la diffusion (broadcast) des poids du modèle aux workers.

#### Étape 1.4 — Chargement des images

```python
images = spark.read.format("binaryFile") \
  .option("pathGlobFilter", "*.jpg") \
  .option("recursiveFileLookup", "true") \
  .load(PATH_Data)
```

Les images sont chargées en **format binaire** directement dans un DataFrame Spark (colonnes : `path`, `modificationTime`, `length`, `content`). Ce format offre de la souplesse pour le prétraitement ultérieur.

Extraction du label depuis le chemin du fichier :

```python
images = images.withColumn('label', element_at(split(images['path'], '/'), -2))
```

Le label correspond au **nom du dossier parent** de l'image (ex: `Apple`, `Banana`...).

#### Étape 1.5 — Préparation du modèle MobileNetV2 (Transfer Learning)

Le modèle **MobileNetV2** (pré-entraîné sur ImageNet) est utilisé pour l'extraction de features :

```python
model = MobileNetV2(weights='imagenet', include_top=True, input_shape=(224, 224, 3))
new_model = Model(inputs=model.input, outputs=model.layers[-2].output)
```

- La **dernière couche de classification** (1000 classes ImageNet) est retirée.
- On conserve **l'avant-dernière couche** qui produit un vecteur de features de dimension **(1, 1, 1280)**.
- Les images originales (100×100 px) sont **redimensionnées à 224×224 px** pour correspondre aux entrées attendues par MobileNetV2.

Les poids du modèle sont **diffusés (broadcast)** depuis le driver vers tous les workers Spark, évitant de recharger le modèle à chaque batch :

```python
brodcast_weights = sc.broadcast(new_model.get_weights())

def model_fn():
    model = MobileNetV2(weights='imagenet', include_top=True, input_shape=(224, 224, 3))
    for layer in model.layers:
        layer.trainable = False
    new_model = Model(inputs=model.input, outputs=model.layers[-2].output)
    new_model.set_weights(brodcast_weights.value)
    return new_model
```

#### Étape 1.6 — Featurisation via Pandas UDF

La chaîne de traitement s'organise en 3 niveaux imbriqués :

```
featurize_udf (Pandas UDF Scalar Iterator)
    └── featurize_series (traitement d'un batch pd.Series)
            └── preprocess (traitement d'une image individuelle)
```

```python
def preprocess(content):
    img = Image.open(io.BytesIO(content)).resize([224, 224])
    arr = img_to_array(img)
    return preprocess_input(arr)   # normalisation spécifique MobileNetV2 [-1, 1]

def featurize_series(model, content_series):
    input = np.stack(content_series.map(preprocess))
    preds = model.predict(input)
    output = [p.flatten() for p in preds]   # aplatit (1,1,1280) → vecteur 1280
    return pd.Series(output)

@pandas_udf('array<float>', PandasUDFType.SCALAR_ITER)
def featurize_udf(content_series_iter):
    model = model_fn()   # chargé une seule fois par worker (pattern Scalar Iterator)
    for content_series in content_series_iter:
        yield featurize_series(model, content_series)
```

Le pattern **`SCALAR_ITER`** est clé : le modèle est chargé **une seule fois par worker** et réutilisé pour tous les batchs successifs, ce qui amortit le coût de chargement sur de grands volumes.

#### Étape 1.7 — Extraction et sauvegarde des features

```python
features_df = images.repartition(20).select(
    col("path"),
    col("label"),
    featurize_udf("content").alias("features")
)
features_df.write.mode("overwrite").parquet(PATH_Result)
```

- `repartition(20)` : distribue les données en 20 partitions pour paralléliser le traitement.
- Les résultats sont écrits au format **Parquet** : colonnaire, compressé, compatible Pandas/Spark.

#### Étape 1.8 — Validation du résultat

```python
df = pd.read_parquet(PATH_Result, engine='pyarrow')
df.head()
df.loc[0, 'features'].shape   # → (1280,) ✓
```

Validation que chaque image produit bien un vecteur de features de dimension 1280.

---

### Partie 2 — Exécution sur le cluster AWS EMR (production)

#### Étape 2.1 — Démarrage de la session Spark

La session est démarrée automatiquement par JupyterHub sur EMR. La commande `%%info` affiche les informations du cluster et les liens vers l'interface Spark UI.

#### Étape 2.2 — Accès aux données S3

```python
PATH        = 's3://tchacoula-p8-data'
PATH_Data   = PATH + '/Test'
PATH_Result = PATH + '/Results'
```

Sur EMR, S3 est monté de façon transparente : les chemins `s3://...` sont utilisés directement comme des chemins locaux, sans téléchargement préalable.

#### Étape 2.3 — Pipeline identique en production

Le code de featurisation (`preprocess`, `featurize_series`, `featurize_udf`, `model_fn`) est **strictement identique** à la version locale. Seuls changent :
- Les chemins (S3 au lieu du système de fichiers local).
- Le nombre de partitions : `repartition(24)` au lieu de 20, adapté aux 24 cœurs du cluster (3 instances × 8 vCPU).
- Le volume de données : 22 819 images au lieu de 330.

#### Étape 2.4 — Réduction de dimension par PCA (MLlib)

Après extraction des features, une **PCA via Spark MLlib** réduit la dimension des vecteurs :

```python
from pyspark.ml.feature import PCA
from pyspark.ml.functions import array_to_vector

# Conversion en vecteur dense Spark
features_df = df.withColumn('features', array_to_vector('features'))
features_df.persist()   # mise en cache pour éviter de recalculer

# PCA initiale sur toutes les composantes (k=1280)
pca = PCA(k=1280, inputCol="features", outputCol="pcaFeatures")
model = pca.fit(features_df)

# Calcul de la variance expliquée cumulée
exp_variance = model.explainedVariance.cumsum()

# Détermination du nombre de composantes pour 99% de variance expliquée
k = np.where(exp_variance < 0.99)[0][-1]

# Application de la PCA finale avec le k optimal
pca = PCA(k=k, inputCol="features", outputCol="pcaFeatures")
model = pca.fit(features_df)
result = model.transform(features_df).select("pcaFeatures")

# Sauvegarde des résultats réduits dans S3
PATH_PCAResult = PATH + '/PCAResults'
result.write.mode("overwrite").parquet(PATH_PCAResult)
```

La PCA conserve les composantes expliquant **moins de 99% de la variance** cumulée, ce qui permet de réduire significativement la dimension des vecteurs tout en préservant l'essentiel de l'information.

---

## Stack technique

| Composant | Technologie |
|-----------|-------------|
| Cloud | Amazon Web Services (AWS) |
| Stockage | Amazon S3 |
| Cluster | Amazon EMR (Elastic MapReduce) |
| Calcul distribué | Apache Spark (PySpark) |
| Framework ML | TensorFlow / Keras |
| Modèle de features | MobileNetV2 (Transfer Learning, ImageNet) |
| Réduction de dimension | PCA — Spark MLlib |
| Format de sortie | Apache Parquet (via PyArrow) |
| Notebook | JupyterHub (sur EMR) |
| Instances | m5.xlarge × 3 (1 primaire + 2 principales) |

---

## Structure du projet

```
.
├── notebook/
│   └── TCHACOULA_Saliou_1_notebook_092024.ipynb   # Notebook PySpark (local + EMR)
├── img/
│   └── mobilenetv2_architecture.png                # Schéma architecture MobileNetV2
├── data/                                           # (local uniquement)
│   ├── Test/                                       # 330 images pour tests locaux
│   └── Results/                                    # Features extraites (Parquet)
└── README.md
```

Sur AWS, les dossiers `data/` correspondent aux préfixes S3 :
```
s3://tchacoula-p8-data/
├── Test/         → 22 819 images JPG
├── Results/      → features MobileNetV2 (Parquet)
└── PCAResults/   → features après PCA (Parquet)
```

---

## Installation et exécution en local

```bash
# Cloner le dépôt
git clone https://github.com/<votre-username>/<votre-repo>.git
cd <votre-repo>

# Installer les dépendances
pip install -r requirements.txt
```

**Exemple de `requirements.txt` :**

```
pyspark
tensorflow
keras
Pillow
numpy
pandas
pyarrow
```

> ⚠️ Java 8 ou 11 doit être installé pour faire tourner PySpark en local.

---

## Configuration du cluster EMR

Lors de la création du cluster EMR, les paramètres suivants ont été utilisés :

- **Logiciels** : Spark, Hadoop, JupyterHub, TensorFlow
- **Instances** : 1 nœud primaire `m5.xlarge` + 2 nœuds principaux `m5.xlarge`
- **Action d'amorçage (bootstrap)** : installation des packages Python (NumPy, Pandas, Pillow...)
- **Persistance** : notebooks sauvegardés automatiquement dans S3
- **Sécurité** : configuration de sécurité + paire de clés EC2 pour l'accès SSH

---

## Auteur

**Saliou TCHACOULA** — Étudiant Data Scientist, OpenClassrooms  
Projet soutenu en septembre 2024.
