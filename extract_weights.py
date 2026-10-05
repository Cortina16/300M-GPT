import torch
from ddp_worker import GPT, vocab_size


CKPT = "checkpoint_step_v2_1200000.pt"
ckpt = torch.load(CKPT, map_location="cpu", weights_only=False)

model = GPT(vocab_size=vocab_size, max_seq_len=1024, d_model=1024, heads=16, num_layers=16)
model.load_state_dict(ckpt["model"])
model = model.to(torch.bfloat16)
torch.save(model.state_dict(), "gpt300m_bf16.pt")
print("saved; params:", sum(p.numel() for p in model.parameters()) / 1e6, "M")