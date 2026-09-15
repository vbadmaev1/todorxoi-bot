# -*- coding: utf-8 -*-
"""
_torch_reference.py — исходная torch-реализация, оставленная ТОЛЬКО для
сверки с NumPy-версией (tools/check_numpy_matches_torch.py).

Боту этот файл не нужен и им не импортируется: torch из зависимостей
убран. Держим его в tools/, чтобы проверку можно было повторить, если
модель переобучат и веса поменяются.
"""

import unicodedata

import torch
import torch.nn as nn
import torch.nn.functional as F

PAD, SOS, EOS, UNK = "<pad>", "<s>", "</s>", "<unk>"


class _CharVocab:
    def __init__(self, itos):
        self.itos = itos
        self.stoi = {ch: i for i, ch in enumerate(itos)}

    def __len__(self):
        return len(self.itos)

    def encode(self, word):
        ids = [self.stoi.get(ch, self.stoi[UNK]) for ch in word]
        return torch.tensor([self.stoi[SOS]] + ids + [self.stoi[EOS]], dtype=torch.long)

    def decode(self, ids):
        chars = []
        for i in ids:
            ch = self.itos[i]
            if ch == EOS:
                break
            if ch in (PAD, SOS):
                continue
            chars.append(ch)
        return "".join(chars)


class _Encoder(nn.Module):
    def __init__(self, vocab_size, emb_dim, hid_dim, pad_idx):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, emb_dim, padding_idx=pad_idx)
        self.gru = nn.GRU(emb_dim, hid_dim, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hid_dim * 2, hid_dim)

    def forward(self, src, src_lens):
        emb = self.emb(src)
        packed = nn.utils.rnn.pack_padded_sequence(
            emb, src_lens.cpu(), batch_first=True, enforce_sorted=False
        )
        packed_out, h = self.gru(packed)
        outputs, _ = nn.utils.rnn.pad_packed_sequence(packed_out, batch_first=True)
        h0 = torch.tanh(self.fc(torch.cat([h[0], h[1]], dim=1)))
        return outputs, h0


class _Attention(nn.Module):
    def __init__(self, hid_dim):
        super().__init__()
        self.attn = nn.Linear(hid_dim * 3, hid_dim)
        self.v = nn.Linear(hid_dim, 1, bias=False)

    def forward(self, dec_hidden, enc_outputs, mask):
        S = enc_outputs.size(1)
        dec_rep = dec_hidden.unsqueeze(1).repeat(1, S, 1)
        energy = torch.tanh(self.attn(torch.cat([dec_rep, enc_outputs], dim=2)))
        scores = self.v(energy).squeeze(2).masked_fill(mask == 0, -1e10)
        return F.softmax(scores, dim=1)


class _Decoder(nn.Module):
    def __init__(self, vocab_size, emb_dim, hid_dim, pad_idx):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, emb_dim, padding_idx=pad_idx)
        self.attention = _Attention(hid_dim)
        self.gru = nn.GRUCell(emb_dim + hid_dim * 2, hid_dim)
        self.out = nn.Linear(hid_dim * 3 + emb_dim, vocab_size)

    def forward(self, input_tok, hidden, enc_outputs, mask):
        emb = self.emb(input_tok)
        attn_weights = self.attention(hidden, enc_outputs, mask)
        context = torch.bmm(attn_weights.unsqueeze(1), enc_outputs).squeeze(1)
        hidden = self.gru(torch.cat([emb, context], dim=1), hidden)
        pred = self.out(torch.cat([hidden, context, emb], dim=1))
        return pred, hidden


def load_torch_reference(pt_path):
    """Возвращает {'translate': функция слово->строка, 'dictionary': ...}."""
    ckpt = torch.load(pt_path, map_location="cpu", weights_only=False)
    src_vocab = _CharVocab(ckpt["src_vocab"])
    tgt_vocab = _CharVocab(ckpt["tgt_vocab"])
    pad_idx = src_vocab.stoi[PAD]
    emb_dim, hid_dim = ckpt["emb_dim"], ckpt["hid_dim"]
    max_len = ckpt.get("max_len", 32)

    enc = _Encoder(len(src_vocab), emb_dim, hid_dim, pad_idx)
    dec = _Decoder(len(tgt_vocab), emb_dim, hid_dim, pad_idx)
    state = ckpt["model_state"]
    enc.load_state_dict({k[len("encoder."):]: v for k, v in state.items()
                         if k.startswith("encoder.")})
    dec.load_state_dict({k[len("decoder."):]: v for k, v in state.items()
                         if k.startswith("decoder.")})
    enc.eval()
    dec.eval()

    sos, eos = tgt_vocab.stoi[SOS], tgt_vocab.stoi[EOS]

    @torch.no_grad()
    def translate(word):
        word = unicodedata.normalize("NFC", word)
        src = src_vocab.encode(word).unsqueeze(0)
        enc_outputs, hidden = enc(src, torch.tensor([src.size(1)]))
        mask = (src != pad_idx)
        token = torch.tensor([sos])
        out = []
        for _ in range(max_len):
            pred, hidden = dec(token, hidden, enc_outputs, mask)
            token = pred.argmax(1)
            tok = int(token.item())
            if tok == eos:
                break
            out.append(tok)
        return tgt_vocab.decode(out)

    return {"translate": translate, "dictionary": ckpt.get("dictionary", {})}
