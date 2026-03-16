# Deep-Symbolic-Audio

Projet de génération symbolique de musique à partir de fichiers MIDI avec deux architectures :

- `lstm`
- `transformer`

Le projet couvre :

- la tokenisation de fichiers MIDI en mode `mono`, `poly` ou `emopia`
- l'entraînement de modèles génératifs
- le fine-tuning d'un checkpoint sur un nouveau vocabulaire
- la génération de nouveaux fichiers MIDI
- une interface Streamlit pour écouter un rendu audio et télécharger le MIDI généré

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Notes :

- pour un rendu audio plus réaliste dans l'UI, il est utile d'avoir un soundfont `.sf2` ou `.sf3`
- sans soundfont, l'application utilise un rendu audio de secours généré en Python

## Structure du projet

- `main.py` : point d'entrée CLI pour `train`, `generate` et `fine-tune`
- `src/tokenizer.py` : création des vocabulaires et des datasets à partir des MIDI
- `src/train.py` : boucles d'entraînement LSTM et Transformer
- `src/fine_tune.py` : adaptation d'un checkpoint à un nouveau vocabulaire
- `src/generate.py` : génération de tokens, conversion MIDI et rendu audio
- `UI/app.py` : interface Streamlit

## Modes de tokenisation

- `mono` : suite de tokens `NOTE_x`, `DUR_x`, `REST_x`
- `poly` : tokens événementiels `NOTE_ON_x`, `NOTE_OFF_x`, `SHIFT_x`
- `emopia` : même principe que `poly`, avec un token de début conditionné par émotion (`START_HAPPY`, `START_SAD`, `START_ANGRY`, `START_RELAXED`)

## Préparation des données

Les fonctions de préparation sont dans `src/tokenizer.py` :

- `create_vocab_and_dataset(...)` pour le mode `mono`
- `create_vocab_and_dataset_polyphonic(...)` pour le mode `poly`
- `create_vocab_and_dataset_emopia(...)` pour le mode `emopia`

Sorties générées :

- dataset(s) `.pt` dans `data/processed/`
- vocabulaires JSON `*_token_to_id.json` et `*_id_to_token.json`

- le bloc `if __name__ == "__main__"` de `src/tokenizer.py` est actuellement configuré pour générer le dataset `emopia`
- pour préparer un autre dataset, il faut modifier/décommenter l'appel voulu dans ce fichier avant de lancer la commande

Exécution :

```bash
python src/tokenizer.py
```

Comportement actuel par mode :

- `mono` : produit un seul fichier dataset, par exemple `data/processed/dataset.pt`
- `poly` : produit un split train/validation, par exemple `dataset_poly_train.pt` et `dataset_poly_val.pt`
- `emopia` : produit aussi un split train/validation, par exemple `dataset_emopia_train.pt` et `dataset_emopia_val.pt`

## Entraînement

Commande générale :

```bash
python main.py train --model <lstm|transformer> --tokenizer-mode <mono|poly> --dataset <dataset.pt>
```

Exemples :

```bash
python main.py train --model lstm --tokenizer-mode mono --dataset data/processed/dataset.pt
```

```bash
python main.py train \
  --model transformer \
  --tokenizer-mode poly \
  --dataset data/processed/dataset_poly_train.pt \
  --val-dataset data/processed/dataset_poly_val.pt
```

Limite actuelle :

- `main.py train` n'expose aujourd'hui que `mono` et `poly` ; pour un vocabulaire émotionnel `emopia`, le flux actuel passe plutôt par le fine-tuning ou par un appel direct aux fonctions Python

Sorties d'entraînement :

- checkpoints dans `models/lstm/` ou `models/transformer/`
- modèle final sous la forme `models/<model>/<model>_<mode>_final.pt`
- pour le Transformer, courbes de loss sauvegardées en PNG dans `models/transformer/`

## Fine-tuning

Le fine-tuning permet de recharger un checkpoint existant et de transférer les poids communs vers un nouveau vocabulaire. C'est utile, par exemple, pour partir d'un modèle `poly` puis l'adapter à `emopia`.

Exemple :

```bash
python main.py fine-tune \
  --model transformer \
  --tokenizer-mode poly \
  --checkpoint models/transformer/transformer_poly_final.pt \
  --old-vocab data/processed/vocab_poly_token_to_id.json \
  --new-vocab data/processed/vocab_emopia_token_to_id.json \
  --dataset data/processed/dataset_emopia_train.pt \
  --val-dataset data/processed/dataset_emopia_val.pt \
  --fine-tune-tag emopia_ft \
  --epochs 30 \
  --lr 1e-4
```

Notes :

- `--old-vocab` doit correspondre au vocabulaire du checkpoint chargé
- `--new-vocab` doit correspondre au nouveau dataset
- `--fine-tune-tag` sert de suffixe pour nommer les checkpoints sauvegardés
- pour un Transformer, un dataset de validation est obligatoire
- un checkpoint issu d'un fine-tuning vers `emopia` peut ensuite être exploité depuis l'UI Streamlit

## Génération MIDI

Commande générale :

```bash
python main.py generate --model <lstm|transformer> --tokenizer-mode <mono|poly>
```

Exemples :

```bash
python main.py generate \
  --model transformer \
  --tokenizer-mode poly \
  --checkpoint models/transformer/transformer_poly_final.pt \
  --output-midi outputs/generated_poly.mid \
  --max-tokens 256 \
  --temperature 0.8
```

Notes :

- si `--checkpoint` n'est pas fourni, `main.py` essaie de retrouver automatiquement le checkpoint final ou le dernier checkpoint disponible
- la CLI de `main.py` n'expose actuellement pas le mode `emopia`
- pour générer à partir d'un checkpoint émotionnel, passe par l'interface Streamlit

## Interface Streamlit

Lancement :

```bash
streamlit run UI/app.py
```

Fonctionnalités principales :

- sélection du modèle et du checkpoint
- sélection du mode de tokens `mono`, `poly` ou `emopia`
- choix de l'émotion pour les checkpoints `emopia`
- génération d'un MIDI et rendu audio WAV
- utilisation automatique ou manuelle d'un soundfont `.sf2/.sf3`
- téléchargement du fichier MIDI généré

Les générations de l'UI sont sauvegardées dans `outputs/ui_generations/`.

## Sorties et chemins utiles

- datasets préprocessés : `data/processed/`
- checkpoints : `models/lstm/` et `models/transformer/`
- MIDI générés en CLI : `outputs/`
- MIDI générés via l'UI : `outputs/ui_generations/`

## Limites actuelles

- la création de dataset n'est pas encore exposée par une vraie CLI avec arguments ; elle passe actuellement par `src/tokenizer.py`
- `main.py train` et `main.py generate` n'exposent aujourd'hui que `mono` et `poly`
- certains exemples de données attendent des dossiers comme `data/midi_mono`, `data/midi_poly` ou `data/midi_emopia`, à préparer manuellement pour l'instant.

## Sources des datasets utilisés

Les datasets bruts ne sont pas inclus dans le projet pour des raisons de poids. Le dépôt exploite uniquement les fichiers MIDI issus de ces jeux de données.

### MAESTRO

- source : [MAESTRO Dataset](https://magenta.tensorflow.org/datasets/maestro)
- article : [Enabling Factorized Piano Music Modeling and Generation with the MAESTRO Dataset](https://openreview.net/forum?id=r1lYRjC9F7)
- usage dans le projet : base MIDI piano pour les pipelines `mono` et `poly`
- dossiers locaux attendus : `data/midi_mono/` ou `data/midi_poly/`

Référence :

```text
Hawthorne, C., Stasyuk, A., Roberts, A., Simon, I., Huang, C.-Z. A.,
Dieleman, S., Elsen, E., Engel, J., and Eck, D. (2019).
Enabling Factorized Piano Music Modeling and Generation with the MAESTRO Dataset.
International Conference on Learning Representations (ICLR).
```

### EMOPIA

- source : [EMOPIA](https://annahung31.github.io/EMOPIA/)
- article : EMOPIA: A Multi-Modal Pop Piano Dataset for Emotion Recognition and Emotion-based Music Generation
- auteurs : Hsiao-Tzu Hung, Joann Ching, Seungheon Doh, Nabin Kim, Juhan Nam, Yi-Hsuan Yang
- publication : ISMIR 2021
- usage dans le projet : apprentissage émotionnel et fine-tuning du vocabulaire `emopia`
- dossier local attendu : `data/midi_emopia/`