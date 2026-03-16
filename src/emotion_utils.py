EMOPIA_EMOTIONS = ("HAPPY", "SAD", "ANGRY", "RELAXED")
EMOPIA_START_TOKENS = {
    f"START_{emotion}": idx for idx, emotion in enumerate(EMOPIA_EMOTIONS)
}
EMOPIA_FILE_PREFIX_TO_EMOTION = {
    "Q1": "HAPPY",
    "Q2": "ANGRY",
    "Q3": "SAD",
    "Q4": "RELAXED",
}


def is_emopia_vocab(token_to_id):
    return all(token in token_to_id for token in EMOPIA_START_TOKENS)
