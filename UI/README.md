# UI

Interface simple pour générer puis écouter un rendu audio des fichiers MIDI produits par le LSTM ou le Transformer.

## Lancer l'interface

Depuis la racine du projet :

streamlit run UI/app.py

## Fonctions

- choix du modèle : `lstm` ou `transformer`
- choix du checkpoint disponible
- génération d'un nouveau morceau
- écoute directe via un rendu WAV
- téléchargement du fichier MIDI généré