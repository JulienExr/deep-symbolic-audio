import torch
import tqdm
import os
from models import MusicLSTM
from dataset import load_dataloaders
from tokenizer import build_vocab
import matplotlib.pyplot as plt
from torch.amp import autocast, GradScaler

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def unpack_batch(batch):
    if len(batch) == 3:
        return batch
    inputs, targets = batch
    return inputs, targets, None


def forward_transformer_batch(model, inputs, emotion=None):
    if getattr(model, "emotion_mode", False):
        if emotion is None:
            raise ValueError("Le modele attend des labels d'emotion, mais le batch n'en contient pas.")
        return model(inputs, emotion)
    return model(inputs)


def get_model_dir(model_name):
    model_dir = os.path.join("models", model_name)
    os.makedirs(model_dir, exist_ok=True)
    return model_dir


def get_checkpoint_path(model_name, tokenizer_mode, epoch):
    model_dir = get_model_dir(model_name)
    return os.path.join(model_dir, f"{model_name}_{tokenizer_mode}_epoch_{epoch}.pt")


def get_final_model_path(model_name, tokenizer_mode):
    model_dir = get_model_dir(model_name)
    return os.path.join(model_dir, f"{model_name}_{tokenizer_mode}_final.pt")

def plot_training_loss(losses, model_name, tokenizer_mode, validation=False):
    plt.figure(figsize=(10, 5))
    plt.plot(losses, label="Training Loss")
    if validation:
        plt.title(f"{model_name.upper()} Training and Validation Loss ({tokenizer_mode} tokenizer)")
        plt.ylabel("Loss")
        plt.legend()
        plt.grid()
        plt.savefig(get_model_dir(model_name) + f"/{model_name}_{tokenizer_mode}_val_loss.png")
    else:
        plt.title(f"{model_name.upper()} Training Loss ({tokenizer_mode} tokenizer)")
        plt.ylabel("Loss")
        plt.legend()
        plt.grid()
        plt.savefig(get_model_dir(model_name) + f"/{model_name}_{tokenizer_mode}_training_loss.png")
    plt.close()

def train_lstm(model, dataloader, num_epochs=10, lr=3e-4, device=device, tokenizer_mode="mono"):
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    criterion = torch.nn.CrossEntropyLoss()
    epoch_losses = []

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0
        progress_bar = tqdm.tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")

        for step, batch in enumerate(progress_bar, start=1):
            inputs, targets, _ = unpack_batch(batch)
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
        if epoch % 5 == 0 or epoch == 0:
            torch.save(model.state_dict(), get_checkpoint_path("lstm", tokenizer_mode, epoch))

    torch.save(model.state_dict(), get_final_model_path("lstm", tokenizer_mode))
    return epoch_losses

def train_transformer(model, dataloader, val_dataloader, num_epochs=10, lr=3e-4, device=device, tokenizer_mode="mono"):
    torch.backends.cudnn.benchmark = True

    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    criterion = torch.nn.CrossEntropyLoss()
    epoch_losses = []
    val_losses = []
    scaler = GradScaler()

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0
        progress_bar = tqdm.tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")

        for step, batch in enumerate(progress_bar, start=1):
            inputs, targets, emotion = unpack_batch(batch)
            inputs, targets = inputs.to(device, non_blocking=True), targets.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with autocast(device_type=device.type):
                if emotion is not None:
                    emotion = emotion.to(device, non_blocking=True)
                outputs = forward_transformer_batch(model, inputs, emotion)
                loss = criterion(outputs.reshape(-1, outputs.size(-1)), targets.reshape(-1))
            
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()
            progress_bar.set_postfix(loss=f"{total_loss / step:.4f}")

        model.eval()
        val_loss = 0.0
        val_progress_bar = tqdm.tqdm(val_dataloader, desc=f"Validation Epoch {epoch+1}/{num_epochs}")

        for step, batch in enumerate(val_progress_bar, start=1):
            inputs, targets, emotion = unpack_batch(batch)
            inputs, targets = inputs.to(device, non_blocking=True), targets.to(device, non_blocking=True)
            if emotion is not None:
                emotion = emotion.to(device, non_blocking=True)

            with torch.no_grad():
                with autocast(device_type=device.type):
                    outputs = forward_transformer_batch(model, inputs, emotion)
                    loss = criterion(outputs.reshape(-1, outputs.size(-1)), targets.reshape(-1))
            val_loss += loss.item()
            val_progress_bar.set_postfix(val_loss=f"{val_loss / step:.4f}")

        epoch_loss = total_loss / len(dataloader)
        epoch_losses.append(epoch_loss)
        val_loss_avg = val_loss / len(val_dataloader)
        val_losses.append(val_loss_avg)

        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}, Validation Loss: {val_loss_avg:.4f}")
        if (epoch + 1) % 5 == 0 or epoch == 0:
            torch.save(model.state_dict(), get_checkpoint_path("transformer", tokenizer_mode, epoch + 1))
            
    torch.save(model.state_dict(), get_final_model_path("transformer", tokenizer_mode))
    plot_training_loss(epoch_losses, "transformer", tokenizer_mode)
    plot_training_loss(val_losses, "transformer", tokenizer_mode, validation=True)
    return epoch_losses

if __name__ == "__main__":
    dataloader = load_dataloaders("data/processed/dataset.pt", batch_size=256)
    vocab, _, _ = build_vocab()
    vocab_size = len(vocab)
    model = MusicLSTM(vocab_size)
    train_lstm(model, dataloader, num_epochs=20, lr=8e-4)
