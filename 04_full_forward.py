"""Episode 04: Full forward pass (no training).
Flow: a -> pre -> h -> f -> logits -> p -> Loss."""
import torch, torch.nn as nn, torch.nn.functional as F

vocab = {"i": 0, "write": 1, "code": 2, "daily": 3}
VOCAB_SIZE, D_MODEL = 4, 2


class MiniLLM(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(VOCAB_SIZE, D_MODEL)   # recipe cards
        self.Wq = nn.Linear(D_MODEL, D_MODEL, bias=False)    # attention Q
        self.Wk = nn.Linear(D_MODEL, D_MODEL, bias=False)    # attention K
        self.Wv = nn.Linear(D_MODEL, D_MODEL, bias=False)    # attention V
        self.W1 = nn.Linear(D_MODEL, D_MODEL, bias=False)    # FFN first thought
        self.W2 = nn.Linear(D_MODEL, D_MODEL, bias=False)    # FFN second thought
        self.W_out = nn.Linear(D_MODEL, VOCAB_SIZE, bias=False)  # 2 -> 4 scores

    def forward(self, ids):
        x = self.embedding(ids)                       # word -> vector
        Q, K, V = self.Wq(x), self.Wk(x), self.Wv(x)
        scores = (Q @ K.transpose(0, 1)) / (D_MODEL ** 0.5)
        attn = F.softmax(scores, dim=-1) @ V          # context-mixed vectors
        a = attn[-1]                                  # only the last token's vector
        pre = self.W1(a)                              # first thought (pre-activation)
        h = F.relu(pre)                               # ReLU gate
        f = self.W2(h)                                # final thought
        return self.W_out(f)                          # logits (4 scores)


if __name__ == "__main__":
    model = MiniLLM()
    inp = torch.tensor([vocab["i"], vocab["write"], vocab["code"]])
    logits = model(inp)
    probs = F.softmax(logits, dim=-1)
    print("logits:", logits.tolist())
    print("probs :", {w: round(probs[i].item(), 3) for w, i in vocab.items()})