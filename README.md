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

- `main.py` : point d'entrée CLI
- `src/project_cli.py` : orchestration des commandes `preprocess`, `train`, `generate` et `fine-tune`
- `src/tokenizer.py` : création des vocabulaires et des datasets à partir des MIDI
- `src/train.py` : boucles d'entraînement LSTM et Transformer
- `src/fine_tune.py` : adaptation d'un checkpoint à un nouveau vocabulaire
- `src/generate.py` : génération de tokens, conversion MIDI et rendu audio
- `src/metrics.py` : rapports JSON/CSV pour l'entraînement, le fine-tuning et la génération
- `UI/app.py` : interface Streamlit

## Modes de tokenisation

- `mono` : suite de tokens `NOTE_x`, `DUR_x`, `REST_x`
- `poly` : tokens événementiels `NOTE_ON_x`, `NOTE_OFF_x`, `SHIFT_x`
- `emopia` : même principe que `poly`, avec un token de début conditionné par émotion (`START_HAPPY`, `START_SAD`, `START_ANGRY`, `START_RELAXED`)

## Préparation des données

La préparation des datasets est maintenant exposée dans la CLI via `main.py preprocess`. Les fonctions sous-jacentes restent dans `src/tokenizer.py`.

Commande générale :

```bash
python main.py preprocess --tokenizer-mode <mono|poly|emopia>
```

Exemples :

```bash
python main.py preprocess --tokenizer-mode mono
```

```bash
python main.py preprocess \
  --tokenizer-mode poly \
  --input-dir data/midi_poly \
  --dataset-output data/processed/dataset_poly.pt \
  --vocab-output data/processed/vocab_poly \
  --max-files 1200 \
  --seed 42
```

```bash
python main.py preprocess \
  --tokenizer-mode emopia \
  --input-dir data/midi_emopia \
  --dataset-output data/processed/dataset_emopia.pt \
  --vocab-output data/processed/vocab_emopia
```

Sorties générées :

- dataset(s) `.pt` dans `data/processed/`
- vocabulaires JSON `*_token_to_id.json` et `*_id_to_token.json`

Comportement actuel par mode :

- `mono` : produit un seul fichier dataset, par exemple `data/processed/dataset.pt`
- `poly` : produit un split train/validation, par exemple `dataset_poly_train.pt` et `dataset_poly_val.pt`
- `emopia` : produit aussi un split train/validation, par exemple `dataset_emopia_train.pt` et `dataset_emopia_val.pt`

Valeurs par défaut utiles :

- `mono` : `input_dir=data/midi_mono`, `dataset_output=data/processed/dataset.pt`, `vocab_output=data/processed/vocab`
- `poly` : `input_dir=data/midi_poly`, `dataset_output=data/processed/dataset_poly.pt`, `vocab_output=data/processed/vocab_poly`
- `emopia` : `input_dir=data/midi_emopia`, `dataset_output=data/processed/dataset_emopia.pt`, `vocab_output=data/processed/vocab_emopia`

Notes :

- `--seq-length` et `--stride` peuvent être surchargés depuis la CLI
- `--seed` contrôle l'ordre des fichiers et le split train/validation pour obtenir un préprocessing reproductible

## Entraînement

Commande générale :

```bash
python main.py train --model <lstm|transformer> --tokenizer-mode <mono|poly|emopia> --dataset <dataset.pt>
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

```bash
python main.py train \
  --model transformer \
  --tokenizer-mode emopia \
  --dataset data/processed/dataset_emopia_train.pt \
  --val-dataset data/processed/dataset_emopia_val.pt \
  --seed 42
```

Sorties d'entraînement :

- checkpoints dans `models/lstm/` ou `models/transformer/`
- modèle final sous la forme `models/<model>/<model>_<mode>_final.pt`
- courbe de loss sauvegardée en PNG dans `models/<model>/`
- rapport JSON `models/<model>/<model>_<mode>_metrics.json`
- historique CSV `models/<model>/<model>_<mode>_history.csv`

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
- un checkpoint issu d'un fine-tuning vers `emopia` peut ensuite être exploité en CLI et dans l'UI Streamlit
- `--seed` permet de rendre le lancement plus reproductible
- un rapport complet est sauvegardé dans `models/<model>/<model>_<fine_tune_tag>_fine_tune_report.json`

## Génération MIDI

Commande générale :

```bash
python main.py generate --model <lstm|transformer> --tokenizer-mode <mono|poly|emopia>
```

Exemples :

```bash
python main.py generate \
  --model transformer \
  --tokenizer-mode poly \
  --checkpoint models/transformer/transformer_poly_final.pt \
  --output-midi outputs/generated_poly.mid \
  --max-tokens 256 \
  --temperature 0.8 \
  --top-k 10 \
  --seed 42
```

```bash
python main.py generate \
  --model transformer \
  --tokenizer-mode emopia \
  --checkpoint models/transformer/transformer_emopia_ft_final.pt \
  --emotion HAPPY \
  --output-midi outputs/generated_emopia_happy.mid
```

Notes :

- si `--checkpoint` n'est pas fourni, `main.py` essaie de retrouver automatiquement le checkpoint final ou le dernier checkpoint disponible
- `--emotion` et `--start-token` permettent de contrôler explicitement le token de départ en mode `emopia`
- un rapport JSON de génération est sauvegardé par défaut à côté du MIDI, par exemple `outputs/generated_poly_metrics.json`

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
- rapports d'entraînement : `models/<model>/*_metrics.json` et `*_history.csv`
- MIDI générés en CLI : `outputs/`
- métriques de génération en CLI : `outputs/*_metrics.json`
- MIDI générés via l'UI : `outputs/ui_generations/`

## Limites actuelles

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
