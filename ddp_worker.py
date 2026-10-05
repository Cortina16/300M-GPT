
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from transformers import get_cosine_schedule_with_warmup
import torch.distributed as dist
import numpy as np
import os
import time
from active_scripts import tokenizer
from contextlib import nullcontext

vocab_to_id, id_to_vocab, bpe_ranks = tokenizer.read_vocab_dict("./vocab_id.vocab")
vocab_size = max(id_to_vocab.keys()) + 1

class MemMapDataset(Dataset):
    def __init__(self, bin_file_path, seq_len=1024):
        file_size = os.path.getsize(bin_file_path)
        num_tokens = file_size // 4
        self.tokens = torch.from_file(
            bin_file_path,
            shared=True,
            size=num_tokens,
            dtype=torch.int32
        )

        self.seq_len = seq_len

    def __len__(self):
        return (len(self.tokens) - 1) // self.seq_len

    def __getitem__(self, idx):
        start_idx = idx * self.seq_len
        end_idx = start_idx + self.seq_len

        x = self.tokens[start_idx:end_idx].clone().long()
        y = self.tokens[start_idx+1:end_idx+1].clone().long()
        return x, y
class TextDataset(Dataset):
    def __init__(self, raw_text, tokenizer, vocab_to_id, bpe_ranks, seq_len = 128):
        print("Tokenizing dataset...")
        self.tokens = torch.tensor(
            tokenizer.encode(raw_text, vocab_to_id, bpe_ranks),
            dtype=torch.int32
        )
        self.seq_len = seq_len
        print(f"Total tokens in dataset: {len(self.tokens):,}")
    def __len__(self,):
        return (len(self.tokens)-1)//self.seq_len

    def __getitem__(self, idx):
        start_idx = idx*self.seq_len
        end_idx = start_idx + self.seq_len
        x = self.tokens[start_idx:end_idx]
        y= self.tokens[start_idx+1:end_idx+1]
        return x ,y


def precompute_rope_freqs(d_k, max_seq_len, theta=10000.0):
    inv_freq = 1.0 / (theta ** (torch.arange(0, d_k, 2).float() / d_k))
    t = torch.arange(max_seq_len, dtype=torch.float32)
    freqs = torch.outer(t, inv_freq)
    cos = torch.cos(freqs)
    sin = torch.sin(freqs)
    return cos, sin

def apply_rope(x, cos, sin):
    seq_len = x.shape[2]

    # Reshape x to treat paired features together: (..., d_k//2, 2)
    x_paired = x.view(*x.shape[:-1], -1, 2)
    x1 = x_paired[..., 0]
    x2 = x_paired[..., 1]

    # Broadasting dimensions for cos/sin: (1, 1, seq_len, d_k//2)
    cos = cos[:seq_len].to(x.device).unsqueeze(0).unsqueeze(1)
    sin = sin[:seq_len].to(x.device).unsqueeze(0).unsqueeze(1)

    # Compute rotated pairs
    o1 = x1 * cos - x2 * sin
    o2 = x1 * sin + x2 * cos

    # Stack along last dim and flatten back to original shape
    return torch.stack([o1, o2], dim=-1).flatten(-2).type_as(x)


class RoPEMQALayer(nn.Module):
    def __init__(self, d_model=768, num_q_heads=12, num_kv_heads=1):
        super().__init__()
        self.num_q_heads = num_q_heads
        self.num_kv_heads = num_kv_heads
        self.d_model = d_model
        self.d_k = d_model // num_q_heads

        self.w_q = nn.Linear(d_model, num_q_heads * self.d_k, bias=False)
        self.w_k = nn.Linear(d_model, num_kv_heads * self.d_k, bias=False)
        self.w_v = nn.Linear(d_model, num_kv_heads * self.d_k, bias=False)
        self.w_o = nn.Linear(num_q_heads * self.d_k, d_model, bias=False)

        self.ffl1 = nn.Linear(d_model, 4*d_model, bias=False)
        self.ffl2 = nn.Linear(4*d_model, d_model, bias=False)
        self.GELU = nn.GELU()
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)


    def _ffn(self, x):
        return self.ffl2(self.GELU(self.ffl1(x)))
    def _attention(self, Q, K, V):
        seq_len = Q.shape[-2]
        scores = Q @ K.transpose(-2, -1) / (self.d_k ** 0.5)

        mask = torch.triu(
            torch.full((seq_len, seq_len), float('-inf'), device=Q.device),
            diagonal=1
        )
        masked_scores = scores+mask
        return nn.functional.softmax(masked_scores, dim=-1) @ V
    def forward(self, x, cos, sin):
        batch_size, seq_len, _ = x.shape

        norm_x = self.ln1(x)
        Q = self.w_q(norm_x).view(batch_size, seq_len, self.num_q_heads, self.d_k).transpose(1, 2)
        K = self.w_k(norm_x).view(batch_size, seq_len, self.num_kv_heads, self.d_k).transpose(1, 2)
        V = self.w_v(norm_x).view(batch_size, seq_len, self.num_kv_heads, self.d_k).transpose(1, 2)

        Q = apply_rope(Q, cos, sin)
        K = apply_rope(K, cos, sin)

        # K = K.expand(batch_size, self.num_q_heads, seq_len, self.d_k)
        # V = V.expand(batch_size, self.num_q_heads, seq_len, self.d_k)

        attn_weights = torch.nn.functional.scaled_dot_product_attention(
            Q, K, V, is_causal=True, enable_gqa=True
        )
        # attn_weights = self._attention(Q, K, V)

        out = attn_weights.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)

        x = x + self.w_o(out)

        x = x + self._ffn(self.ln2(x))

        return x


class GPT(nn.Module):
    def __init__(self, vocab_size=vocab_size, d_model=768, max_seq_len=1024, heads=12, num_layers=4):
        super().__init__()
        self.max_seq_len=max_seq_len
        self.w_embedding = nn.Embedding(vocab_size, d_model)

        cos, sin = precompute_rope_freqs(d_model // heads, max_seq_len)
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)

        # self.pos_embedding = nn.Embedding(max_seq_len, d_model)
        self.layers = nn.ModuleList([
            RoPEMQALayer(d_model=d_model, num_q_heads=heads, num_kv_heads=1) for _ in range(num_layers)
        ])

        self.ln_f = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        self.w_embedding.weight = self.lm_head.weight
        def _init(m):
            if isinstance(m, (nn.Linear, nn.Embedding)):
                nn.init.normal_(m.weight, mean=0.0, std=0.02)
        self.apply(_init)

        resid_std = 0.02 / (2 * num_layers) ** 0.5
        for layer in self.layers:
            nn.init.normal_(layer.w_o.weight, std=resid_std)
            nn.init.normal_(layer.ffl2.weight, std=resid_std)
    def forward(self, input_tokens):
        batch_size, seq_len = input_tokens.shape
        x = self.w_embedding(input_tokens)
        cos = self.cos[:seq_len]
        sin = self.sin[:seq_len]
        for layer in self.layers:
            x = layer(x, cos, sin)
        x = self.ln_f(x)
        logits = self.lm_head(x)
        return logits

class SkipSampler(torch.utils.data.Sampler):
    def __init__(self, base, skip):
        self.base, self.skip = base, skip
    def __iter__(self):
        it = iter(self.base)
        for _ in range(self.skip):
            next(it)
        return it
    def __len__(self):
        return len(self.base) - self.skip

@torch.no_grad()
def generate(model, prompt_ids, max_new_tokens=50, temperature=1.0):
    model.eval()
    if prompt_ids.ndim == 1:
        prompt_ids = prompt_ids.unsqueeze(0)
    generated = prompt_ids.clone()

    for _ in range(max_new_tokens):
        input_tokens = generated[:, -model.max_seq_len:]
        logits = model(input_tokens)
        next_token = logits[:, -1, :] / temperature

        probs = nn.functional.softmax(next_token, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        generated = torch.cat([generated, next_token], dim=1)
    return generated.squeeze(0)

@torch.no_grad()
def generate_sample_text(model, prompt_text="The history of the", max_new_tokens=50, device='cuda'):
    model.eval()
    input_ids = tokenizer.encode(prompt_text, vocab_to_id, bpe_ranks)
    x = torch.tensor([input_ids], dtype=torch.long, device=device)

    for _ in range(max_new_tokens):
        x_cond = x[:, -1024:] # Context window clip
        logits = model(x_cond)
        next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
        x = torch.cat((x, next_token), dim=1)

    model.train()
    return tokenizer.decode(x[0].tolist(), id_to_vocab)

def format_time(seconds):
    """Converts seconds to HH:MM:SS format."""
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    return f"{h:02d}:{m:02d}:{s:02d}" if h > 0 else f"{m:02d}:{s:02d}"

def setup_ddp(local_rank, world_size):
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
    torch.cuda.set_device(local_rank)

    dist.init_process_group("nccl")

def cleanup_ddp():
    dist.destroy_process_group()

@torch.no_grad()
def generate_sample(model, prompt_tokens, max_new_tokens=50, temperature=0.7, top_p=0.9, device='cuda'):
    raw_model = model.module if hasattr(model, 'module') else model
    raw_model.eval()
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

    x = prompt_tokens.clone().unsqueeze(0) # (1, seq_len)

    for _ in range(max_new_tokens):
        x_cond = x[:, -raw_model.max_seq_len:]

        with torch.amp.autocast('cuda', dtype=dtype):
            logits = model(x_cond)

        next_token_logits = logits[:, -1, :] / temperature

        sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
        cumulative_probs = torch.cumsum(torch.nn.functional.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above threshold
        sorted_indices_to_remove = cumulative_probs > top_p
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
        next_token_logits[indices_to_remove] = -float('Inf')

        # Sample from filtered distribution
        probs = torch.nn.functional.softmax(next_token_logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)

        x = torch.cat((x, next_token), dim=1)

    raw_model.train()
    return tokenizer.decode(x[0].tolist(), id_to_vocab)



def train_worker(local_rank, world_size):
    setup_ddp(local_rank, world_size)
    device = torch.device(f"cuda:{local_rank}")
    is_main_process = (local_rank == 0)

    torch.cuda.empty_cache()
    ckpt = torch.load("checkpoint_step_v2_380000.pt", map_location="cpu")
    start_step = ckpt["step"] + 1
    BATCH_SIZE_PER_GPU = 6
    GRAD_ACCUM_STEPS = 8
    SEQ_LEN = 1024
    EPOCHS = 1
    LEARNING_RATE = 3e-4
    LOG_INTERVAL = 50
    dataset = MemMapDataset("../dataset.bin", seq_len=SEQ_LEN)
    assert (
        dataset.tokens.max().item() < vocab_size
    ), f"Dataset token ID {dataset.tokens.max().item()} exceeds model vocab size {vocab_size}!"
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=local_rank, shuffle=True)
    dataloader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE_PER_GPU,
        sampler=SkipSampler(sampler, start_step * BATCH_SIZE_PER_GPU),
        num_workers=3,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2
    )
    raw_model = GPT(
        vocab_size=vocab_size,
        max_seq_len=SEQ_LEN,
        d_model=1024,
        heads=16,
        num_layers=16
    ).to(device)
    decay = [p for p in raw_model.parameters() if p.ndim >= 2]
    no_decay = [p for p in raw_model.parameters() if p.ndim < 2]
    optimizer = torch.optim.AdamW([{"params": decay, "weight_decay": 0.1},
     {"params": no_decay, "weight_decay": 0.0}],
    lr=LEARNING_RATE, betas=(0.9, 0.95), eps=1e-8,)
    WARMUP_STEPS = 600  # Ramps LR from 0 to 3e-4 over first 2000 steps
    TOTAL_OPTIMIZER_STEPS = ((len(dataloader)+start_step) // GRAD_ACCUM_STEPS) * EPOCHS

    scheduler = get_cosine_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=WARMUP_STEPS,
        num_training_steps=TOTAL_OPTIMIZER_STEPS,
        # min_lr_ratio=0.1 # Decays down to 10% of peak LR (3e-5) at the end
    )
    raw_model.load_state_dict(ckpt["model"])
    optimizer.load_state_dict(ckpt["optimizer"])
    scheduler.load_state_dict(ckpt["scheduler"])
    model = DDP(raw_model, device_ids=[local_rank])

    model = torch.compile(model)

    del ckpt

    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    scaler = torch.amp.GradScaler('cuda', enabled=(dtype == torch.float16))
    loss_fn = nn.CrossEntropyLoss()

    total_steps = len(dataloader) + start_step
    start_training_time = time.time()
    average_loss = 0
    model.train()
    print(start_step)
    print(total_steps)
    print(scheduler.get_last_lr()[0])
    for epoch in range(EPOCHS):
        sampler.set_epoch(epoch)
        total_loss = 0.0
        epoch_start_time = time.time()
        step_start_time = time.time()
        optimizer.zero_grad(set_to_none = True)
        if is_main_process:
            log_f = open("../metrics.csv", "a", buffering=1)
            if log_f.tell() == 0:
                log_f.write("micro_step,opt_step,tokens,loss_avg,grad_norm_last,grad_norm_max,lr,tok_per_s,vram_gb\n")
        last_gn, max_gn = float("nan"), 0.0
        for step, (x_batch, y_batch) in enumerate(dataloader, start=start_step):
            x_batch = x_batch.to(device, non_blocking=True)
            y_batch = y_batch.to(device, non_blocking=True)

            is_accumulating = (step + 1) % GRAD_ACCUM_STEPS != 0
            context = model.no_sync() if is_accumulating else nullcontext()
            with context:
                with torch.amp.autocast('cuda', dtype=dtype):
                    logits = model(x_batch)
                    unscaled_loss = loss_fn(
                        logits.view(-1, vocab_size),
                        y_batch.view(-1)
                    )
                    average_loss += unscaled_loss.item()
                    loss = unscaled_loss / GRAD_ACCUM_STEPS

                if dtype == torch.float16:
                    scaler.scale(loss).backward()
                else:
                    loss.backward()

            total_loss += unscaled_loss.item()
            # * GRAD_ACCUM_STEPS
            if not is_accumulating:
                if dtype == torch.float16:
                    scaler.unscale_(optimizer)
                    gn = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0).item()
                    last_gn, max_gn = gn, max(max_gn, gn)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    gn = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0).item()
                    last_gn, max_gn = gn, max(max_gn, gn)
                    optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)

            if is_main_process and (step + 1) % 50 == 0:
                elapsed_interval = time.time() - step_start_time
                steps_per_sec = LOG_INTERVAL / elapsed_interval
                tokens_per_sec = steps_per_sec * (BATCH_SIZE_PER_GPU * world_size) * SEQ_LEN

                steps_remaining = total_steps - (step + 1)
                eta_epoch = steps_remaining / steps_per_sec if steps_per_sec > 0 else 0

                vram_info = ""
                if torch.cuda.is_available():
                    vram_mb = torch.cuda.max_memory_allocated(device) / (1024 ** 2)
                    vram_info = f" | Peak VRAM: {vram_mb:.0f}MB"
                    allocated = torch.cuda.memory_allocated(device) / (1024 ** 2)
                    reserved = torch.cuda.memory_reserved(device) / (1024 ** 2)
                print(

                    f"Epoch [{epoch+1}/{EPOCHS}] | "
                    f"Active Tensors: {allocated:.0f}MB | Reserved Pool: {reserved:.0f}MB"
                    f"Step [{step+1}/{total_steps}] | "
                    f"Loss: {unscaled_loss.item():.4f} | "
                    f"Average Loss: {average_loss/50:.4f}"
                    f"Speed: {steps_per_sec:.2f} st/s ({tokens_per_sec:.0f} tok/s) | "
                    f"ETA Epoch: {format_time(eta_epoch)}"
                    f"{vram_info}"
                )
                tokens = (step + 1) * BATCH_SIZE_PER_GPU * world_size * SEQ_LEN
                log_f.write(f"{step+1},{(step+1)//GRAD_ACCUM_STEPS},{tokens},{average_loss/LOG_INTERVAL:.4f},"
                            f"{last_gn:.4f},{max_gn:.4f},{scheduler.get_last_lr()[0]:.3e},"
                            f"{tokens_per_sec:.0f},{vram_mb/1024:.2f}\n")
                max_gn = 0.0
                average_loss = 0
                step_start_time = time.time()
                predicted_ids = logits[0].argmax(dim=-1).tolist()
                target_text_clean = tokenizer.decode(y_batch[0].tolist(), id_to_vocab).replace('\n', ' ')
                sample_pred_clean = tokenizer.decode(predicted_ids, id_to_vocab).replace('\n', ' ')

                print(f"TARGET BATCH SAMPLE :\n\"{target_text_clean[:150]}...\"")
                print(f"MODEL PREDICTION    :\n\"{sample_pred_clean[:150]}...\"\n")
            if is_main_process and (step + 1) % 500 == 0:
                with torch.amp.autocast('cuda', dtype=dtype):
                    print(f"SAMPLE AUTOREGRESSIVE STRING: \n\"{generate_sample_text(raw_model).replace('\n', ' ')}\"")

            if is_main_process and (step+1) % 5000 == 0:
                checkpoint = {
                    'model': raw_model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'epoch': epoch,
                    'step': step,
                    'scheduler': scheduler.state_dict()
                }
                torch.save(checkpoint, f"checkpoint_step_v2_{step+1}.pt")
            if step == 20:
                # torch.cuda.empty_cache()
                pass
        if is_main_process:
            epoch_time = time.time() - epoch_start_time
            avg_loss = total_loss / len(dataloader)
            print(f"--- Epoch {epoch+1} Complete | Average Loss: {avg_loss:.4f} ---\n")
    if is_main_process:
        total_training_time = time.time() - start_training_time
        print(f"🎉 Training Completed in {format_time(total_training_time)}!")
    cleanup_ddp()

if __name__ == "__main__":

    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    train_worker(local_rank, world_size)
