# Deep-Symbolic-Audio

Projet de génération symbolique de musique à partir de fichiers MIDI.

## Installation

```bash
pip install -r requirements.txt
```

## Pipeline

1. tokenisation des MIDI
2. entraînement d'un `lstm` ou d'un `transformer`
3. génération de nouveaux fichiers MIDI
4. écoute via l'interface Streamlit

## Commandes utiles

### Créer un dataset

```bash
python src/tokenizer.py
```

### Entraîner un modèle

```bash
python main.py train --model transformer --tokenizer-mode poly --dataset data/processed/dataset_poly_train.pt --val-dataset data/processed/dataset_poly_val.pt
```

### Générer un MIDI

```bash
python main.py generate --model transformer --tokenizer-mode poly
```

### Lancer l'UI

```bash
streamlit run UI/app.py
```

## Notes

- modes de tokenisation disponibles : `mono`, `poly`
- modèles disponibles : `lstm`, `transformer`
- les poids sont sauvegardés dans `models/lstm/` et `models/transformer/`