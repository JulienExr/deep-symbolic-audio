import math
import os
import matplotlib.pyplot as plt
import torch
import tqdm
from torch.amp import autocast, GradScaler

from dataio.dataset import load_dataloaders
from metrics import count_model_parameters, build_training_report, write_training_report
from modeling.architectures import MusicLSTM
from symbolic.tokenizer import build_vocab

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def resolve_warmup_steps(total_steps, warmup_ratio):
    if not 0.0 <= warmup_ratio < 1.0:
        raise ValueError(f"warmup_ratio doit etre dans [0, 1), recu: {warmup_ratio}")
    if total_steps <= 1 or warmup_ratio == 0.0:
        return 0
    return min(total_steps - 1, max(1, int(round(total_steps * warmup_ratio))))


class WarmupCosineScheduler:
    def __init__(self, optimizer, total_steps, warmup_steps=0, min_lr_ratio=0.0):
        if total_steps <= 0:
            raise ValueError(f"total_steps doit etre > 0, recu: {total_steps}")
        if not 0.0 <= min_lr_ratio <= 1.0:
            raise ValueError(f"min_lr_ratio doit etre dans [0, 1], recu: {min_lr_ratio}")

        self.optimizer = optimizer
        self.total_steps = total_steps
        self.warmup_steps = min(max(0, warmup_steps), total_steps - 1 if total_steps > 1 else 0)
        self.min_lr_ratio = min_lr_ratio
        self.base_lrs = [group["lr"] for group in optimizer.param_groups]
        self.step_index = 0
        self._apply_lr(self._lr_scale(self.step_index))

    def _lr_scale(self, step_index):
        if self.warmup_steps > 0 and step_index < self.warmup_steps:
            return float(step_index + 1) / float(self.warmup_steps)

        decay_span = self.total_steps - self.warmup_steps - 1
        if decay_span <= 0:
            return 1.0

        progress = min(1.0, max(0.0, (step_index - self.warmup_steps) / decay_span))
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return self.min_lr_ratio + (1.0 - self.min_lr_ratio) * cosine

    def _apply_lr(self, scale):
        for base_lr, group in zip(self.base_lrs, self.optimizer.param_groups):
            group["lr"] = base_lr * scale

    def step(self):
        if self.step_index < self.total_steps - 1:
            self.step_index += 1
        self._apply_lr(self._lr_scale(self.step_index))

    def get_last_lr(self):
        return [group["lr"] for group in self.optimizer.param_groups]


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
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs, _ = model(inputs)
            loss = criterion(outputs.reshape(-1, outputs.size(-1)), targets.reshape(-1))
            total_loss += loss.item()

    return total_loss / len(dataloader)


def evaluate_transformer(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0.0
    progress_bar = tqdm.tqdm(dataloader, desc="Validation", leave=False)

    for step, (inputs, targets) in enumerate(progress_bar, start=1):
        inputs, targets = inputs.to(device, non_blocking=True), targets.to(device, non_blocking=True)

        with torch.no_grad():
            with autocast(device_type=device.type):
                outputs = model(inputs)
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

        for step, (inputs, targets) in enumerate(progress_bar, start=1):
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
    model_name="transformer",
    num_epochs=10,
    lr=3e-4,
    warmup_ratio=0.1,
    min_lr_ratio=0.0,
    device=device,
    tokenizer_mode="mono",
    metrics_metadata=None,
):
    torch.backends.cudnn.benchmark = True

    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    total_steps = max(1, num_epochs * len(dataloader))
    warmup_steps = resolve_warmup_steps(total_steps, warmup_ratio)
    scheduler = WarmupCosineScheduler(
        optimizer,
        total_steps=total_steps,
        warmup_steps=warmup_steps,
        min_lr_ratio=min_lr_ratio,
    )
    criterion = torch.nn.CrossEntropyLoss()
    epoch_losses = []
    val_losses = []
    scaler = GradScaler()
    checkpoint_paths = []
    metrics_metadata = dict(metrics_metadata or {})
    metrics_metadata.update(
        {
            "scheduler": "warmup_cosine_decay",
            "base_learning_rate": lr,
            "warmup_ratio": warmup_ratio,
            "warmup_steps": warmup_steps,
            "total_optimizer_steps": total_steps,
            "min_lr_ratio": min_lr_ratio,
        }
    )

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0
        progress_bar = tqdm.tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")

        for step, (inputs, targets) in enumerate(progress_bar, start=1):
            inputs, targets = inputs.to(device, non_blocking=True), targets.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with autocast(device_type=device.type):
                outputs = model(inputs)
                loss = criterion(outputs.reshape(-1, outputs.size(-1)), targets.reshape(-1))
            
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            total_loss += loss.item()
            progress_bar.set_postfix(loss=f"{total_loss / step:.4f}", lr=f"{scheduler.get_last_lr()[0]:.2e}")

        epoch_loss = total_loss / len(dataloader)
        epoch_losses.append(epoch_loss)
        val_loss_avg = evaluate_transformer(model, val_dataloader, criterion, device)
        val_losses.append(val_loss_avg)

        print(
            f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}, "
            f"Validation Loss: {val_loss_avg:.4f}, LR: {scheduler.get_last_lr()[0]:.2e}"
        )
        if (epoch + 1) % 5 == 0 or epoch == 0:
            checkpoint_path = get_checkpoint_path(model_name, tokenizer_mode, epoch + 1)
            torch.save(model.state_dict(), checkpoint_path)
            checkpoint_paths.append(checkpoint_path)
            
    final_model_path = get_final_model_path(model_name, tokenizer_mode)
    torch.save(model.state_dict(), final_model_path)
    plot_training_loss(epoch_losses, val_losses, model_name, tokenizer_mode)
    metrics_metadata["final_learning_rate"] = scheduler.get_last_lr()[0]

    report = build_training_report(
        model_name=model_name,
        tokenizer_mode=tokenizer_mode,
        train_losses=epoch_losses,
        val_losses=val_losses,
        metadata=metrics_metadata,
        checkpoint_paths=checkpoint_paths,
        final_model_path=final_model_path,
        parameter_stats=count_model_parameters(model),
    )
    write_training_report(
        json_path=get_metrics_json_path(model_name, tokenizer_mode),
        csv_path=get_metrics_csv_path(model_name, tokenizer_mode),
        report=report,
    )
    return report
