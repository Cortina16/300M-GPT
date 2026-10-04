import sys, torch
from ddp_worker import GPT, vocab_size, id_to_vocab, vocab_to_id, bpe_ranks
import tokenizer

device = "cuda" if torch.cuda.is_available() else "cpu"
model = GPT(vocab_size=vocab_size, max_seq_len=1024, d_model=1024, heads=16, num_layers=16)
model.load_state_dict(torch.load("gpt300m_bf16.pt", map_location="cpu"))
model = model.to(device).eval()

@torch.no_grad()
def generate(prompt, max_new_tokens=100, temperature=0.8, top_k=50, top_p=0.9, rep_penalty=1.1):
    ids = tokenizer.encode(prompt, vocab_to_id, bpe_ranks)
    x = torch.tensor([ids], dtype=torch.long, device=device)
    for _ in range(max_new_tokens):
        logits = model(x[:, -1024:])[0, -1].float()
        for t in set(x[0].tolist()):
            logits[t] = logits[t] / rep_penalty if logits[t] > 0 else logits[t] * rep_penalty
        logits = logits / temperature
        if top_k:
            kth = torch.topk(logits, top_k).values[-1]
            logits[logits < kth] = float("-inf")
        probs = torch.softmax(logits, dim=-1)
        sp, si = torch.sort(probs, descending=True)
        cum = torch.cumsum(sp, dim=-1)
        sp[(cum - sp) > top_p] = 0
        probs = torch.zeros_like(probs).scatter_(0, si, sp)
        probs /= probs.sum()
        nxt = torch.multinomial(probs, 1)
        x = torch.cat([x, nxt.view(1,1)], dim=1)
    return tokenizer.decode(x[0].tolist(), id_to_vocab)
if __name__ == "__main__":
    prompt = " ".join(sys.argv[1:]) or "The history of the united states"
    print(generate(prompt))