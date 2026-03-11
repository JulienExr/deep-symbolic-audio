import torch
from src.tokenizer import create_vocab_and_dataset
from src.dataset import load_dataloaders
from src.train import train_lstm
from src.generate import generate_lstm, tokens_to_midi
from src.tokenizer import build_vocab
from model import MusicLSTM
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if __name__ == "__main__":
    # create_vocab_and_dataset("data/midi_mono", "data/processed/dataset.pt", "data/processed/vocab", max_files=1000, seq_length=128, stride=8)
    dataloader = load_dataloaders("data/processed/dataset.pt", batch_size=64)
    vocab, token_to_id, id_to_token = build_vocab()
    vocab_size = len(vocab)
    model = MusicLSTM(vocab_size)
    model.load_state_dict(torch.load("models/lstm_final.pt", map_location=device))
    model.to(device)
    start_token_id = token_to_id["START"]