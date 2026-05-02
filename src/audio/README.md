# Pipeline audio

Ce dossier contient le pipeline audio direct du projet :

```text
WAV -> segments -> tokens EnCodec -> Transformer causal -> tokens EnCodec -> WAV
```

## Modules

- `audio_io.py` : lecture, inspection et ecriture audio via `ffmpeg`/`ffprobe`.
- `prepare_audio.py` : conversion mono 32 kHz, normalisation et decoupage en segments de 5 secondes.
- `tokenize_encodec.py` : tokenisation des segments avec `facebook/encodec_32khz`.
- `dataset.py` : chargement des fichiers `.pt`, entrelacement des codebooks et fenetres d'entrainement.
- `model.py` : Transformer causal pour les tokens audio.
- `train.py` : entrainement, checkpoints, courbes de loss et generations intermediaires.
- `generate.py` : generation autoregressive et decodage EnCodec vers WAV.
- `find_vocab_size.py` : inspection rapide de la plage de tokens sauvegardes.

## Commandes utiles

```bash
python src/audio/prepare_audio.py
```

```bash
python src/audio/tokenize_encodec.py \
  --input_dir data/audio/prepared \
  --output_dir data/audio/tokenized \
  --limit 10000
```

```bash
python src/audio/train.py \
  --tokens_dir data/audio/tokenized/tokens \
  --vocab_size 2048 \
  --seq_len 512 \
  --stride 256 \
  --batch_size 4 \
  --epochs 55
```

```bash
python src/audio/generate.py \
  --checkpoint checkpoints/audio/best.pt \
  --output_wav outputs/generated.wav \
  --start_tokens_file data/audio/tokenized/tokens/example.pt
```
