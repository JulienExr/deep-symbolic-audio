import json
from pathlib import Path

import torch

from dataset import load_dataloaders
from emotion_utils import is_emopia_vocab
from models import build_music_lstm, build_music_transformer
from train import train_lstm, train_transformer


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_vocab_mapping(vocab_path):
    with open(vocab_path, "r") as file:
        token_to_id = json.load(file)
    return token_to_id


def infer_model_name_from_state_dict(state_dict):
    if "head.weight" in state_dict:
        return "transformer"
    if "fc.weight" in state_dict:
        return "lstm"
    raise ValueError("Impossible de déduire l'architecture du checkpoint.")


def get_embedding_key(model_name):
    if model_name in {"lstm", "transformer"}:
        return "embedding.weight"
    raise ValueError(f"Architecture non supportée: {model_name}")


def get_output_keys(model_name):
    if model_name == "transformer":
        return "head.weight", "head.bias"
    if model_name == "lstm":
        return "fc.weight", "fc.bias"
    raise ValueError(f"Architecture non supportée: {model_name}")


def build_model_for_fine_tuning(model_name, vocab_size, emotion_mode=False):
    if model_name == "transformer":
        return build_music_transformer(vocab_size, emotion_mode=emotion_mode)
    if model_name == "lstm":
        return build_music_lstm(vocab_size)
    raise ValueError(f"Architecture non supportée: {model_name}")


def _copy_common_token_weights(old_tensor, new_tensor, old_token_to_id, new_token_to_id):
    copied_tokens = 0
    for token, new_idx in new_token_to_id.items():
        old_idx = old_token_to_id.get(token)
        if old_idx is None:
            continue
        new_tensor[new_idx].copy_(old_tensor[old_idx])
        copied_tokens += 1
    return copied_tokens


def resize_and_load_state_dict(model, checkpoint_state, old_token_to_id, new_token_to_id, model_name=None):
    if model_name is None:
        model_name = infer_model_name_from_state_dict(checkpoint_state)

    new_state = model.state_dict()
    embedding_key = get_embedding_key(model_name)
    output_weight_key, output_bias_key = get_output_keys(model_name)

    skipped_keys = {embedding_key, output_weight_key, output_bias_key}
    compatible_state = {
        key: value
        for key, value in checkpoint_state.items()
        if key in new_state and key not in skipped_keys and new_state[key].shape == value.shape
    }
    new_state.update(compatible_state)

    missing_or_reshaped = [
        key
        for key, value in checkpoint_state.items()
        if key in new_state and key not in compatible_state and key not in skipped_keys
    ]

    old_embedding = checkpoint_state[embedding_key]
    new_embedding = new_state[embedding_key]
    copied_embedding_tokens = _copy_common_token_weights(
        old_embedding,
        new_embedding,
        old_token_to_id,
        new_token_to_id,
    )

    old_output_weight = checkpoint_state[output_weight_key]
    new_output_weight = new_state[output_weight_key]
    copied_output_tokens = _copy_common_token_weights(
        old_output_weight,
        new_output_weight,
        old_token_to_id,
        new_token_to_id,
    )

    old_output_bias = checkpoint_state[output_bias_key]
    new_output_bias = new_state[output_bias_key]
    _copy_common_token_weights(
        old_output_bias.unsqueeze(-1),
        new_output_bias.unsqueeze(-1),
        old_token_to_id,
        new_token_to_id,
    )

    model.load_state_dict(new_state)

    return {
        "model_name": model_name,
        "old_vocab_size": len(old_token_to_id),
        "new_vocab_size": len(new_token_to_id),
        "copied_embedding_tokens": copied_embedding_tokens,
        "copied_output_tokens": copied_output_tokens,
        "loaded_shared_layers": len(compatible_state),
        "skipped_reshaped_layers": missing_or_reshaped,
        "new_tokens": sorted(set(new_token_to_id) - set(old_token_to_id)),
        "removed_tokens": sorted(set(old_token_to_id) - set(new_token_to_id)),
    }


def create_fine_tune_model(
    checkpoint_path,
    old_vocab_path,
    new_vocab_path,
    model_name=None,
    device=device,
):
    checkpoint_path = Path(checkpoint_path)
    old_vocab_path = Path(old_vocab_path)
    new_vocab_path = Path(new_vocab_path)

    checkpoint_state = torch.load(checkpoint_path, map_location="cpu")
    old_token_to_id = load_vocab_mapping(old_vocab_path)
    new_token_to_id = load_vocab_mapping(new_vocab_path)

    if model_name is None:
        model_name = infer_model_name_from_state_dict(checkpoint_state)

    emotion_mode = "emotion_embedding.weight" in checkpoint_state or is_emopia_vocab(new_token_to_id)
    model = build_model_for_fine_tuning(model_name, len(new_token_to_id), emotion_mode=emotion_mode)
    transfer_report = resize_and_load_state_dict(
        model=model,
        checkpoint_state=checkpoint_state,
        old_token_to_id=old_token_to_id,
        new_token_to_id=new_token_to_id,
        model_name=model_name,
    )
    model.to(device)
    return model, transfer_report


def fine_tune_model(
    checkpoint_path,
    old_vocab_path,
    new_vocab_path,
    train_dataset_path,
    model_name=None,
    val_dataset_path=None,
    batch_size=256,
    num_epochs=10,
    lr=None,
    tokenizer_mode="fine_tune",
    device=device,
):
    model, transfer_report = create_fine_tune_model(
        checkpoint_path=checkpoint_path,
        old_vocab_path=old_vocab_path,
        new_vocab_path=new_vocab_path,
        model_name=model_name,
        device=device,
    )

    require_emotions = getattr(model, "emotion_mode", False)
    dataloader = load_dataloaders(
        train_dataset_path,
        batch_size=batch_size,
        require_emotions=require_emotions,
    )

    effective_model_name = transfer_report["model_name"]
    if effective_model_name == "transformer":
        if val_dataset_path is None:
            raise ValueError("Un dataset de validation est requis pour fine-tuner le transformer.")

        val_dataloader = load_dataloaders(
            val_dataset_path,
            batch_size=batch_size,
            shuffle=False,
            require_emotions=require_emotions,
        )
        losses = train_transformer(
            model,
            dataloader,
            val_dataloader,
            num_epochs=num_epochs,
            lr=lr if lr is not None else 3e-4,
            device=device,
            tokenizer_mode=tokenizer_mode,
        )
    else:
        losses = train_lstm(
            model,
            dataloader,
            num_epochs=num_epochs,
            lr=lr if lr is not None else 8e-4,
            device=device,
            tokenizer_mode=tokenizer_mode,
        )

    return model, transfer_report, losses


if __name__ == "__main__":
    root_dir = Path(__file__).resolve().parents[1]
    _, report, _ = fine_tune_model(
        checkpoint_path=root_dir / "models/transformer/transformer_poly_final.pt",
        old_vocab_path=root_dir / "data/processed/vocab_poly_token_to_id.json",
        new_vocab_path=root_dir / "data/processed/vocab_emopia_token_to_id.json",
        train_dataset_path=root_dir / "data/processed/dataset_emopia_train.pt",
        val_dataset_path=root_dir / "data/processed/dataset_emopia_val.pt",
        model_name="transformer",
        num_epochs=30,
        lr=1e-4,
    )
    print(f"Modele charge: {report['model_name']}")
    print(f"Ancien vocabulaire: {report['old_vocab_size']}")
    print(f"Nouveau vocabulaire: {report['new_vocab_size']}")
    print(f"Tokens copies dans l'embedding: {report['copied_embedding_tokens']}")
    print(f"Tokens copies dans la sortie: {report['copied_output_tokens']}")
