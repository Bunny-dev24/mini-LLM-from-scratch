"""Episode 05: Training loop — forward -> loss -> backward -> update."""
import torch, torch.nn as nn, torch.nn.functional as F
from importlib import import_module

# Import the model from episode 04 (or copy the class here directly)
MiniLLM = import_module("04_full_forward").MiniLLM
vocab = import_module("04_full_forward").vocab
id_to_word = {v: k for k, v in vocab.items()}

torch.manual_seed(0)                    # makes the output reproducible
model = MiniLLM()
inp = torch.tensor([vocab["i"], vocab["write"], vocab["code"]])
tgt = torch.tensor(vocab["daily"])       # the correct answer WE provide
opt = torch.optim.SGD(model.parameters(), lr=0.1)  # w = w - lr*grad

for e in range(200):
    logits = model(inp)                                            # forward
    loss = F.cross_entropy(logits.unsqueeze(0), tgt.unsqueeze(0))  # softmax + loss
    opt.zero_grad()                                                # clear old grads
    loss.backward()                                               # chain rule
    opt.step()                                                    # tune weights
    if e % 40 == 0 or e == 199:
        p = F.softmax(logits, dim=-1)
        print(f"Epoch {e:3d} | Loss={loss.item():.4f} | p(daily)={p[3].item():.3f}")

with torch.no_grad():
    p = F.softmax(model(inp), dim=-1)
    print("\nFinal:", {w: f"{p[i].item()*100:.1f}%" for w, i in vocab.items()})
    print("Guess ->", id_to_word[torch.argmax(p).item()])