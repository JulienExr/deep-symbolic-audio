import torch
import torch.nn as nn


class MusicLSTM(nn.Module):
    def __init__(self, vocab_size, emb_dim=128, hidden_dim=256, num_layers=2, dropout=0.1):
        super(MusicLSTM, self).__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim)
        self.lstm = nn.LSTM(emb_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x, hidden=None):
        x = self.embedding(x)
        output, hidden = self.lstm(x, hidden)
        output = self.fc(output)
        return output, hidden


class MusicTransformer(nn.Module):
    def __init__(self, vocab_size, d_model, n_heads, n_layers, d_ff, dropout, max_len, pad_token_id):
        super(MusicTransformer, self).__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.max_len = max_len
        self.pad_token_id = pad_token_id
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=pad_token_id)
        self.pos_embedding = nn.Embedding(max_len, d_model)
        self.dropout = nn.Dropout(dropout)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model,
                                                    nhead=n_heads,
                                                    dim_feedforward=d_ff,
                                                    dropout=dropout,
                                                    batch_first=True,
                                                    activation='gelu',
                                                    norm_first=True
                                                )
        
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size)

    def _causal_mask(self, seq_len: int, device: torch.device):
        return torch.triu(torch.ones(seq_len, seq_len, device=device, dtype=torch.bool), diagonal=1)
    
    def forward(self, x):

        batch_size, seq_len = x.shape   
        if seq_len > self.max_len:
            raise ValueError(f"Sequence length {seq_len} exceeds model's maximum length {self.max_len}")
        
        positions = torch.arange(seq_len, device=x.device).unsqueeze(0).expand(batch_size, -1)

        token_emb = self.embedding(x)
        pos = self.pos_embedding(positions)
        h = self.dropout(token_emb + pos)

        causal_mask = self._causal_mask(seq_len, x.device)
        pad_mask = (x == self.pad_token_id)

        h = self.transformer(h, mask=causal_mask, src_key_padding_mask=pad_mask)
        h = self.ln_f(h)
        logits = self.head(h)
        return logits

def build_music_lstm(vocab_size, emb_dim=128, hidden_dim=256, num_layers=2, dropout=0.1):
    return MusicLSTM(vocab_size, emb_dim, hidden_dim, num_layers, dropout)

def build_music_transformer(vocab_size, d_model=384, n_heads=6, n_layers=6, d_ff=1536, dropout=0.1, max_len=512, pad_token_id=0):
    return MusicTransformer(vocab_size, d_model, n_heads, n_layers, d_ff, dropout, max_len, pad_token_id)
