import torch
import tqdm
from model import MusicLSTM
from dataset import load_dataloaders
from tokenizer import build_vocab

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train_lstm(model, dataloader, num_epochs=10, lr=3e-4, device=device):
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    criterion = torch.nn.CrossEntropyLoss()
    epoch_losses = []

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0
        progress_bar = tqdm.tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")

        for step, (inputs, targets) in enumerate(progress_bar, start=1):
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs, _ = model(inputs)
            loss = criterion(outputs.reshape(-1, outputs.size(-1)), targets.reshape(-1))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

            progress_bar.set_postfix(loss=f"{total_loss / step:.4f}")

        epoch_loss = total_loss / len(dataloader)
        epoch_losses.append(epoch_loss)
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}")

    torch.save(model.state_dict(), "models/lstm_final.pt")
    return epoch_losses



if __name__ == "__main__":
    dataloader = load_dataloaders("data/processed/dataset.pt", batch_size=256)
    vocab, _, _ = build_vocab()
    vocab_size = len(vocab)
    model = MusicLSTM(vocab_size)
    train_lstm(model, dataloader, num_epochs=20, lr=8e-4)