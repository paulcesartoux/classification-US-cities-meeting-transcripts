# Meeting Task Categorizer — ML Project

> **Université Paris 1 Panthéon-Sorbonne · MIAGE**  
> Tambone Quentin · Toux Paul-Cesar · Baumgartner Etienne · Marques Hugo

---

## Description

Ce projet de Machine Learning s'inscrit dans un système plus large de gestion automatique de réunions. L'objectif est de **catégoriser automatiquement des tâches** extraites de transcriptions de réunions en trois catégories métier :

| Catégorie | Description |
|---|---|
| `Document_Writing` | Rédaction de rapports, ordonnances, résolutions, etc. |
| `Meeting_Planning` | Planification et organisation de réunions |
| `Administrative_Communication` | Emails officiels, communiqués, mémos |

---

## Pipeline global

```
Enregistrement → Transcription → Extraction → Création de tâches → Personnalisation → Exécution
                                     ↑
                           (focus de ce projet)
```

---

## Dataset

### Source
Aucun dataset adapté n'existant, nous l'avons **annoté from scratch** à partir du dataset de transcriptions de réunions municipales :
- [MeetingBank Transcript — HuggingFace](https://meetingbank.github.io/)
- Transcriptions volumineuses : **~2775 mots en moyenne** (écart-type : ~5610)

### Génération avec ChatGPT
Nous avons utilisé `gpt-4o-mini` via LangChain pour :
1. Associer chaque entrée à l'une des trois catégories
2. Transformer les transcriptions longues en descriptions courtes et exploitables

Le script a tourné **toute une nuit** pour un coût d'environ **5$**.

### Statistiques du dataset final
- **9 246 entrées** au total
- 3 catégories, **déséquilibrées** (Document_Writing surreprésenté)

---

## Préparation des données

### Nettoyage
- Suppression des stopwords (NLTK)
- Mise en minuscules
- Reformatage : `Category` / `Description`

### Rééquilibrage (4 variantes)

| Dataset | Méthode | Description |
|---|---|---|
| `df_full_VAE` | Sur-échantillonnage VAE | +2197 Meeting Planning, +1619 Admin. Com. |
| `df_full_SMOTE` | Sur-échantillonnage SMOTE | +2137 Meeting Planning, +1619 Admin. Com. |
| `df_downsampled_VAE` | Sur/Sous-échantillonnage VAE | -1619 Doc. Writing, +518 Meeting Planning |
| `df_all_downsampled` | Sous-échantillonnage aléatoire | -2137 Doc. Writing, -518 Admin. Com. |

---

## Modèles

### Modèles principaux
- **KNN** — K-Nearest Neighbors
- **Random Forest** — agrégation d'arbres de décision

### Modèles secondaires
- Logistic Regression
- Perceptron
- Linear SVC
- MLP Classifier

### Optimisation des hyperparamètres
Recherche aléatoire via `RandomizedSearchCV` (scoring : `f1_weighted`, `cv=3`).

---

## Résultats

### Meilleurs modèles individuels (sur `df_full_VAE`)

| Modèle | F1 Score (test) |
|---|---|
| **MLP Classifier** | **0.855** |
| Linear SVC | 0.850 |
| Logistic Regression | 0.848 |
| Random Forest | 0.804 |
| KNN | 0.717 |

### Métriques détaillées — MLP (df_full_VAE)

|  | Precision | Recall | F1-score |
|---|---|---|---|
| Train | 0.92 | 0.92 | 0.92 |
| Test | 0.86 | 0.86 | 0.85 |

### Ensembles

| Ensemble | F1 Score (test) |
|---|---|
| KNN + Random Forest | 0.72 – 0.82 |
| MLP + LogReg + SVC | 0.85 |
| KNN + RFC (soft voting) | 0.76 |

> Les ensembles n'ont pas surpassé les meilleurs modèles individuels.

---

## Limitations

- Dataset partiellement généré par un LLM (ChatGPT) → qualité difficile à vérifier
- Volume de données limité
- Déséquilibre initial important entre les classes
- Domaine très spécifique (réunions municipales américaines)

---

## Stack technique

- `scikit-learn` — modèles ML (KNN, RF, SVC, MLP, LogReg, Perceptron), SMOTE, RandomizedSearchCV
- `pandas` / `numpy` — manipulation des données
- `sentence-transformers` — embeddings SBERT (`all-MiniLM-L6-v2`), MPNet, BGE
- `transformers` + `torch` — embeddings BERT
- `tensorflow` / `keras` — VAE custom pour sur-échantillonnage dans l'espace des embeddings
- `langchain-openai` + `gpt-4o-mini` — génération du dataset labellisé

---

## Structure du projet

```
.
├── Projet/
│   └── dataset/
│       ├── training_data.csv         # Données d'entraînement (source)
│       └── test_data.csv             # Données de test (source)
├── dataset_full_VAE.csv              # Sur-échantillonnage VAE (toutes classes à égalité)
├── dataset_downsample_VAE.csv        # Sous/Sur-échantillonnage VAE
├── dataset_full_SMOTE.csv            # Sur-échantillonnage SMOTE
├── dataset_all_downsampled.csv       # Sous-échantillonnage aléatoire
├── synthetic_embeddings.csv          # Embeddings synthétiques générés par le VAE
└── notebook.ipynb                    # Notebook principal (pipeline complet)
```

### Fonctions clés du notebook

| Fonction | Rôle |
|---|---|
| `get_text_embeddings(texts, encodeur)` | Génère des embeddings (tfidf, sbert, mpnet, bge, bert, word2vec, fasttext) |
| `embed_and_concat(df, ...)` | Embarque les descriptions et les concatène au DataFrame |
| `generate_synthetic_data_vae_from_embeddings(...)` | VAE : génère des embeddings synthétiques pour rééquilibrer |
| `balance_classes(df, n_samples)` | Sous-échantillonnage de la classe dominante |
| `downsample_all_classes(df, target_size)` | Sous-échantillonnage de toutes les classes |
| `run_randomized_search_on_dataset(...)` | RandomizedSearchCV sur tous les modèles |
| `train_all_models_on_all_datasets(...)` | Entraîne tous les modèles sur tous les datasets |
| `compute_scores(df_train, df_test, model)` | Calcule et affiche les métriques (precision, recall, F1) |

---

## Auteurs

| Nom |
|---|
| Tambone Quentin |
| Toux Paul-Cesar |
| Baumgartner Etienne |
| Marques Hugo |

---

*Projet réalisé dans le cadre du cursus MIAGE — Université Paris 1 Panthéon-Sorbonne*