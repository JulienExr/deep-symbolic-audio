import os
import matplotlib.pyplot as plt
import torch
import tqdm
from torch.amp import autocast, GradScaler

from dataset import load_dataloaders
from metrics import count_model_parameters, build_training_report, write_training_report
from models import MusicLSTM
from tokenizer import build_vocab

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

def get_metrics_json_path(model_name, tokenizer_mode):
    model_dir = get_model_dir(model_name)
    return os.path.join(model_dir, f"{model_name}_{tokenizer_mode}_metrics.json")


def get_metrics_csv_path(model_name, tokenizer_mode):
    model_dir = get_model_dir(model_name)
    return os.path.join(model_dir, f"{model_name}_{tokenizer_mode}_history.csv")


def plot_training_loss(train_losses, val_losses, model_name, tokenizer_mode):
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label="Training Loss")
    if val_losses:
        plt.plot(val_losses, label="Validation Loss")
        title = f"{model_name.upper()} Training and Validation Loss ({tokenizer_mode} tokenizer)"
    else:
        title = f"{model_name.upper()} Training Loss ({tokenizer_mode} tokenizer)"
    plt.title(title)
    plt.ylabel("Loss")
    plt.xlabel("Epoch")
    plt.legend()
    plt.grid()
    plt.savefig(get_model_dir(model_name) + f"/{model_name}_{tokenizer_mode}_loss.png")
    plt.close()

def evaluate_lstm(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0.0

    with torch.no_grad():
        for batch in dataloader:
            inputs, targets, _ = unpack_batch(batch)
            inputs, targets = inputs.to(device), targets.to(device)
            outputs, _ = model(inputs)
            loss = criterion(outputs.reshape(-1, outputs.size(-1)), targets.reshape(-1))
            total_loss += loss.item()

    return total_loss / len(dataloader)


def evaluate_transformer(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0.0
    progress_bar = tqdm.tqdm(dataloader, desc="Validation", leave=False)

    for step, batch in enumerate(progress_bar, start=1):
        inputs, targets, emotion = unpack_batch(batch)
        inputs, targets = inputs.to(device, non_blocking=True), targets.to(device, non_blocking=True)
        if emotion is not None:
            emotion = emotion.to(device, non_blocking=True)

        with torch.no_grad():
            with autocast(device_type=device.type):
                outputs = forward_transformer_batch(model, inputs, emotion)
                loss = criterion(outputs.reshape(-1, outputs.size(-1)), targets.reshape(-1))
        total_loss += loss.item()
        progress_bar.set_postfix(val_loss=f"{total_loss / step:.4f}")

    return total_loss / len(dataloader)


def train_lstm(
    model,
    dataloader,
    val_dataloader=None,
    num_epochs=10,
    lr=3e-4,
    device=device,
    tokenizer_mode="mono",
    metrics_metadata=None,
):
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    criterion = torch.nn.CrossEntropyLoss()
    train_losses = []
    val_losses = []
    checkpoint_paths = []

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

        train_loss = total_loss / len(dataloader)
        train_losses.append(train_loss)

        val_loss = None
        if val_dataloader is not None:
            val_loss = evaluate_lstm(model, val_dataloader, criterion, device)
            val_losses.append(val_loss)

        if val_loss is not None:
            print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}")
        else:
            print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {train_loss:.4f}")

        if epoch % 5 == 0 or epoch == 0:
            checkpoint_path = get_checkpoint_path("lstm", tokenizer_mode, epoch)
            torch.save(model.state_dict(), checkpoint_path)
            checkpoint_paths.append(checkpoint_path)

    final_model_path = get_final_model_path("lstm", tokenizer_mode)
    torch.save(model.state_dict(), final_model_path)
    plot_training_loss(train_losses, val_losses, "lstm", tokenizer_mode)

    report = build_training_report(
        model_name="lstm",
        tokenizer_mode=tokenizer_mode,
        train_losses=train_losses,
        val_losses=val_losses,
        metadata=metrics_metadata,
        checkpoint_paths=checkpoint_paths,
        final_model_path=final_model_path,
        parameter_stats=count_model_parameters(model),
    )
    write_training_report(
        json_path=get_metrics_json_path("lstm", tokenizer_mode),
        csv_path=get_metrics_csv_path("lstm", tokenizer_mode),
        report=report,
    )
    return report

def train_transformer(
    model,
    dataloader,
    val_dataloader,
    num_epochs=10,
    lr=3e-4,
    device=device,
    tokenizer_mode="mono",
    metrics_metadata=None,
):
    torch.backends.cudnn.benchmark = True

    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    criterion = torch.nn.CrossEntropyLoss()
    epoch_losses = []
    val_losses = []
    scaler = GradScaler()
    checkpoint_paths = []

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

        epoch_loss = total_loss / len(dataloader)
        epoch_losses.append(epoch_loss)
        val_loss_avg = evaluate_transformer(model, val_dataloader, criterion, device)
        val_losses.append(val_loss_avg)

        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}, Validation Loss: {val_loss_avg:.4f}")
        if (epoch + 1) % 5 == 0 or epoch == 0:
            checkpoint_path = get_checkpoint_path("transformer", tokenizer_mode, epoch + 1)
            torch.save(model.state_dict(), checkpoint_path)
            checkpoint_paths.append(checkpoint_path)
            
    final_model_path = get_final_model_path("transformer", tokenizer_mode)
    torch.save(model.state_dict(), final_model_path)
    plot_training_loss(epoch_losses, val_losses, "transformer", tokenizer_mode)

    report = build_training_report(
        model_name="transformer",
        tokenizer_mode=tokenizer_mode,
        train_losses=epoch_losses,
        val_losses=val_losses,
        metadata=metrics_metadata,
        checkpoint_paths=checkpoint_paths,
        final_model_path=final_model_path,
        parameter_stats=count_model_parameters(model),
    )
    write_training_report(
        json_path=get_metrics_json_path("transformer", tokenizer_mode),
        csv_path=get_metrics_csv_path("transformer", tokenizer_mode),
        report=report,
    )
    return report

if __name__ == "__main__":
    dataloader = load_dataloaders("data/processed/dataset.pt", batch_size=256)
    vocab, _, _ = build_vocab()
    vocab_size = len(vocab)
    model = MusicLSTM(vocab_size)
    train_lstm(model, dataloader, num_epochs=20, lr=8e-4)
