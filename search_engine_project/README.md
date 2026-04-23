# AI-Based Multimedia Search Engine with Web Interface

Ce projet universitaire implemente un moteur de recherche multimedia en Python capable de rechercher dans un corpus de documents texte et d'afficher des images liees aux requetes. Il repose sur un index inverse, une recherche vectorielle et booleenne, un agent IA pour ameliorer le traitement des requetes, ainsi qu'une interface web developpee avec Streamlit.

## Fonctionnalites principales

- Pretraitement des documents texte
- Construction d'un index inverse
- Recherche vectorielle avec TF-IDF et similarite cosinus
- Recherche booleenne avec `AND`, `OR`, `NOT`
- Agent IA pour les suggestions et l'amelioration des requetes
- Affichage de documents pertinents et d'images associees
- Evaluation des resultats avec `Precision` et `Recall`
- Interface web sombre avec Streamlit

## Structure actuelle du projet

```text
search_engine_project/
|-- documents/
|-- images/
|-- ai_agent.py
|-- boolean_model.py
|-- evaluation.py
|-- index.json
|-- indexer.py
|-- MoteurRecherche.py
|-- vector_model.py
|-- web_app.py
`-- README.md
```

## Role des principaux fichiers

- `indexer.py` : lit les documents, applique le pretraitement et construit l'index inverse.
- `vector_model.py` : effectue le classement des documents avec TF-IDF et similarite cosinus.
- `boolean_model.py` : gere la recherche booleenne.
- `MoteurRecherche.py` : regroupe la logique principale du moteur de recherche.
- `ai_agent.py` : fournit les suggestions, l'expansion de requete et l'explication des resultats.
- `evaluation.py` : calcule les metriques d'evaluation comme la precision et le rappel.
- `web_app.py` : interface web Streamlit integrant la recherche et l'agent IA.
- `index.json` : fichier d'index genere a partir du corpus.

## Construire l'index

Depuis le dossier `search_engine_project`, executez :

```bash
python indexer.py
```

Le programme analyse les fichiers du dossier `documents/` puis enregistre l'index inverse dans `index.json`.

## Lancer l'interface web

Depuis le dossier `search_engine_project`, executez :

```bash
streamlit run web_app.py
```

L'application web s'ouvre dans le navigateur et permet :

- de saisir une requete,
- de laisser l'agent IA guider la strategie de recherche,
- d'afficher les resultats texte et les images liees,
- de visualiser les metriques `Precision` et `Recall` lorsqu'une reference d'evaluation existe.

## Exemples de requetes

- `machine learning`
- `information retrieval`
- `computer vision`
- `audio processing`
- `boolean model`
- `multimedia search`
- `learning AND neural`
- `retrieval OR ranking`
- `image NOT audio`

## Evaluation

Le projet inclut un module d'evaluation base sur des ensembles de pertinence predefinis pour certaines requetes. Les metriques principales sont :

- `Precision`
- `Recall`

Une courbe Precision-Recall peut egalement etre generee pour analyser le comportement du moteur sur les resultats classes.

## Remarque pedagogique

Le stemming utilise dans ce projet reste volontairement simple et base sur des regles locales. Ce choix permet de garder une implementation claire, pedagogique et adaptee a un contexte academique de decouverte des techniques de recherche d'information.
