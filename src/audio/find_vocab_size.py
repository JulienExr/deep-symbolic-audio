from pathlib import Path
import torch

token_dir = Path("./data/audio/tokenized/tokens")

max_token = -1
min_token = 10**18

for path in token_dir.glob("*.pt"):
    payload = torch.load(path, map_location="cpu")
    codes = payload["codes"]   # [K, T]
    max_token = max(max_token, int(codes.max().item()))
    min_token = min(min_token, int(codes.min().item()))

print("min_token =", min_token)
print("max_token =", max_token)
print("vocab_size =", max_token + 1)