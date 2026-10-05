

import math, time, torch
from torch import nn
from torch.utils.data import DataLoader
from ddp_worker import GPT, MemMapDataset, vocab_size
import tokenizer
from ddp_worker import id_to_vocab, vocab_to_id, bpe_ranks

device = "cuda"
SEQ = 512
BS = 12

ds = MemMapDataset("../dataset.bin", seq_len=SEQ)
dl = DataLoader(ds, batch_size=BS, shuffle=True, num_workers=2, drop_last=True)

model = GPT(vocab_size=vocab_size, d_model=512, max_seq_len=SEQ, heads=8, num_layers=4).to(device)
loss_fn = nn.CrossEntropyLoss()

def run(x, y):
    with torch.autocast("cuda", dtype=torch.bfloat16):
        logits = model(x)
        return loss_fn(logits.view(-1, vocab_size).float(), y.view(-1))

# ---- 1. initial loss: should be ~ln(vocab) ----
x, y = next(iter(dl))
x, y = x.to(device), y.to(device)
with torch.no_grad():
    print(f"init loss {run(x, y).item():.3f}   (expect ~{math.log(vocab_size):.2f})")

# ---- 2. overfit one batch: should go to ~0 ----
opt = torch.optim.AdamW(model.parameters(), lr=1e-3, betas=(0.9, 0.95), weight_decay=0.0)
for i in range(200):
    loss = run(x, y)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step(); opt.zero_grad(set_to_none=True)
    if i % 25 == 0:
        print(f"overfit step {i}: {loss.item():.4f}")

# ---- 3. short real run from fresh weights ----
model = GPT(vocab_size=vocab_size, d_model=512, max_seq_len=SEQ, heads=8, num_layers=4).to(device)
decay = [p for p in model.parameters() if p.ndim >= 2]
no_decay = [p for p in model.parameters() if p.ndim < 2]
opt = torch.optim.AdamW(
    [{"params": decay, "weight_decay": 0.1}, {"params": no_decay, "weight_decay": 0.0}],
    lr=1e-3, betas=(0.9, 0.95))
STEPS, WARM = 2000, 100
sched = torch.optim.lr_scheduler.LambdaLR(
    opt, lambda s: min(1, (s + 1) / WARM) * 0.5 * (1 + math.cos(math.pi * min(s, STEPS) / STEPS)))

running, t0 = [], time.time()
it = iter(dl)
for step in range(STEPS):
    x, y = next(it)
    x, y = x.to(device), y.to(device)
    loss = run(x, y)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step(); sched.step(); opt.zero_grad(set_to_none=True)
    running.append(loss.item())
    if (step + 1) % 50 == 0:
        print(f"step {step+1}: avg loss {sum(running[-50:])/50:.3f}  ({time.time()-t0:.0f}s)")
    if (step + 1) % 500 == 0:
        ids = torch.tensor([tokenizer.encode("The history of", vocab_to_id, bpe_ranks)], device=device)
        model.eval()
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            for _ in range(40):
                nxt = model(ids[:, -SEQ:])[:, -1].argmax(-1, keepdim=True)
                ids = torch.cat([ids, nxt], 1)
        model.train()
        print("SAMPLE:", tokenizer.decode(ids[0].tolist(), id_to_vocab).replace("\n", " "))
