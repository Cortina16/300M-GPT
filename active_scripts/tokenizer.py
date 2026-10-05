# def stream_texts(file_paths):
#     decoder = json.JSONDecoder()
#     for file_path in file_paths:
#         with open(file_path, "r", encoding='utf-8', errors="ignore") as f:
#             content = f.read()
#         idx = 0
#         length = len(content)
#         while idx < length:
#             idx = content.find("{", idx)
#             if idx == -1:
#                 break
#             try:
#                 obj, end_idx = decoder.raw_decode(content, idx)
#                 if isinstance(obj, dict) and "text" in obj:
#                     yield obj["text"]
#                 idx = end_idx
#             except json.JSONDecodeError:
#                 idx += 1
# all_files = glob.glob("text/*.jsonl")
# from tqdm import tqdm
# with open("output.txt", 'w') as f:
#     for text in tqdm(stream_texts(all_files), unit=" docs"):
#         f.write(f"{text}\n")
import collections
import glob
import heapq
import json
import multiprocessing as mp
import os
import re
import struct


# ---------------------------------------------------------
# Byte <-> Unicode mapping (GPT-2 / Hugging Face standard)
# ---------------------------------------------------------
def bytes_to_unicode():
    bs = (
            list(range(ord("!"), ord("~") + 1))
            + list(range(ord("¡"), ord("¬") + 1))
            + list(range(ord("®"), ord("ÿ") + 1))
    )
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return dict(zip(bs, [chr(i) for i in cs]))


byte_encoder = bytes_to_unicode()
byte_decoder = {v: k for k, v in byte_encoder.items()}

SPLIT_PATTERN = re.compile(r"""'s|'t|'re|'ve|'m|'ll|'d| ?\w+| ?\S+|\s+""")


# ---------------------------------------------------------
# Parallel Pre-tokenization Worker
# ---------------------------------------------------------
def _process_file_chunk(args):
    file_path, start_byte, end_byte = args
    counts = collections.Counter()
    with open(file_path, "rb") as f:
        f.seek(start_byte)
        if start_byte != 0:
            f.readline()  # Skip partial line

        while f.tell() < end_byte:
            line = f.readline()
            if not line:
                break
            try:
                text = line.decode("utf-8", errors="ignore")
                matches = SPLIT_PATTERN.findall(text)
                for w in matches:
                    if w.strip():
                        # Map chunk directly to unicode-byte token tuple
                        tokens = tuple(byte_encoder[b] for b in w.encode("utf-8"))
                        counts[tokens] += 1
            except Exception:
                continue
    return counts


def parallel_count_words(corpus_path, num_workers=32):
    file_size = os.path.getsize(corpus_path)
    chunk_size = max(file_size // num_workers, 1)

    tasks = []
    for i in range(num_workers):
        start = i * chunk_size
        end = file_size if i == num_workers - 1 else (i + 1) * chunk_size
        tasks.append((corpus_path, start, end))

    print(f"Tokenizing and counting across {num_workers} processes...")
    total_counts = collections.Counter()
    with mp.Pool(processes=num_workers) as pool:
        for partial_counts in pool.imap_unordered(_process_file_chunk, tasks):
            total_counts.update(partial_counts)

    return total_counts


# ---------------------------------------------------------
# Doubly Linked List Node for O(1) Local Merging
# ---------------------------------------------------------
class Node:
    __slots__ = ("token", "prev", "next")

    def __init__(self, token):
        self.token = token
        self.prev = None
        self.next = None


# ---------------------------------------------------------
# Optimized BPE Training (Indexed Heap / FastBPE)
# ---------------------------------------------------------
def train_bpe_fast(corpus_path, target_vocab_size=128000, num_workers=32):
    word_counts = parallel_count_words(corpus_path, num_workers=num_workers)
    MIN_FREQ = 3
    word_counts = {w: c for w, c in word_counts.items() if c >= MIN_FREQ}
    # 1. Build doubly linked lists for each unique pre-tokenized word
    words = []
    freqs = []
    pair_freqs = collections.defaultdict(int)
    where_to_find = collections.defaultdict(set)

    print("Building token graphs and index...")
    for word_id, (token_tuple, freq) in enumerate(word_counts.items()):
        freqs.append(freq)
        head = Node(token_tuple[0])
        curr = head
        for t in token_tuple[1:]:
            nxt = Node(t)
            curr.next = nxt
            nxt.prev = curr
            curr = nxt
        words.append(head)

        curr = head
        while curr.next is not None:
            pair = (curr.token, curr.next.token)
            pair_freqs[pair] += freq
            where_to_find[pair].add(word_id)
            curr = curr.next

    # 2. Priority Queue: (-freq, pair)
    heap = [(-count, pair) for pair, count in pair_freqs.items()]
    heapq.heapify(heap)

    # Base vocabulary (256 individual bytes)
    vocab_to_id = {char: idx for idx, char in enumerate(byte_encoder.values())}
    merges = []
    num_merges = target_vocab_size - len(vocab_to_id)

    print(f"Learning {num_merges} merge operations...")

    for step in range(num_merges):
        # Pop valid top pair
        best_pair = None
        while heap:
            neg_count, candidate = heapq.heappop(heap)
            if pair_freqs.get(candidate, 0) == -neg_count and -neg_count > 0:
                best_pair = candidate
                break

        if not best_pair:
            print(f"No more pairs to merge. Stopping at step {step}.")
            break

        first, second = best_pair
        new_token = first + second
        merges.append(best_pair)
        vocab_to_id[new_token] = len(vocab_to_id)

        del pair_freqs[best_pair]
        target_words = list(where_to_find[best_pair])
        del where_to_find[best_pair]

        # Local updates only on affected word graphs
        for word_id in target_words:
            curr = words[word_id]
            freq = freqs[word_id]

            while curr is not None:
                if curr.token == first and curr.next is not None and curr.next.token == second:
                    # Remove adjacent pairs being destroyed
                    if curr.prev is not None:
                        left_pair = (curr.prev.token, curr.token)
                        pair_freqs[left_pair] -= freq
                        where_to_find[left_pair].discard(word_id)
                        heapq.heappush(heap, (-pair_freqs[left_pair], left_pair))

                    if curr.next.next is not None:
                        right_pair = (curr.next.token, curr.next.next.token)
                        pair_freqs[right_pair] -= freq
                        where_to_find[right_pair].discard(word_id)
                        heapq.heappush(heap, (-pair_freqs[right_pair], right_pair))

                    # Merge the nodes
                    curr.token = new_token
                    curr.next = curr.next.next
                    if curr.next is not None:
                        curr.next.prev = curr

                    # Add new adjacent pairs formed by the merge
                    if curr.prev is not None:
                        new_left = (curr.prev.token, curr.token)
                        pair_freqs[new_left] += freq
                        where_to_find[new_left].add(word_id)
                        heapq.heappush(heap, (-pair_freqs[new_left], new_left))

                    if curr.next is not None:
                        new_right = (curr.token, curr.next.token)
                        pair_freqs[new_right] += freq
                        where_to_find[new_right].add(word_id)
                        heapq.heappush(heap, (-pair_freqs[new_right], new_right))
                else:
                    curr = curr.next

        if (step + 1) % 5000 == 0:
            print(f"Merges completed: {step + 1}/{num_merges}")
            heap = [(-count, pair) for pair, count in pair_freqs.items() if count > 0]
            heapq.heapify(heap)

    id_to_vocab = {v: k for k, v in vocab_to_id.items()}
    bpe_ranks = {pair: idx for idx, pair in enumerate(merges)}
    return vocab_to_id, id_to_vocab, bpe_ranks


# ---------------------------------------------------------
# Inference: Encode & Decode
# ---------------------------------------------------------
def bpe_merge_word(word_tokens, bpe_ranks):
    if len(word_tokens) <= 1:
        return word_tokens

    tokens = list(word_tokens)
    while len(tokens) > 1:
        pairs = [(tokens[i], tokens[i + 1]) for i in range(len(tokens) - 1)]
        min_pair = min(pairs, key=lambda p: bpe_ranks.get(p, float("inf")))
        if min_pair not in bpe_ranks:
            break

        first, second = min_pair
        new_tokens = []
        i = 0
        while i < len(tokens):
            if i < len(tokens) - 1 and tokens[i] == first and tokens[i + 1] == second:
                new_tokens.append(first + second)
                i += 2
            else:
                new_tokens.append(tokens[i])
                i += 1
        tokens = new_tokens
    return tokens


def encode(text, vocab_to_id, bpe_ranks):
    chunks = SPLIT_PATTERN.findall(text)
    token_ids = []
    for chunk in chunks:
        raw_bytes = chunk.encode("utf-8")
        byte_tokens = [byte_encoder[b] for b in raw_bytes]
        merged_tokens = bpe_merge_word(byte_tokens, bpe_ranks)
        token_ids.extend([vocab_to_id[token] for token in merged_tokens])
    return token_ids


def decode(token_ids, id_to_vocab):
    tokens = [
        id_to_vocab[idx]
        for idx in token_ids
        if idx in id_to_vocab  # Prevents KeyError on missing gap IDs
    ]

    byte_sequence = bytes([
        byte_decoder[char]
        for token in tokens
        for char in token
        if char in byte_decoder
    ])

    return byte_sequence.decode("utf-8", errors="replace")

def save_vocab_dict(dict, bpe_ranks, path):
    sorted_merges = sorted(bpe_ranks.keys(), key=lambda p: bpe_ranks[p])

    with open(path, 'wb') as f:
        f.write(struct.pack('>I', len(dict)))
        for token, token_id in dict.items():
            token_bytes = token.encode('utf-8') if isinstance(token, str) else token
            # token_len = len(token_bytes)
            f.write(struct.pack('>IH', token_id, len(token_bytes)))
            f.write(token_bytes)
        f.write(struct.pack(">I", len(sorted_merges)))
        for first, second in sorted_merges:
            fb = first.encode("utf-8")
            sb = second.encode("utf-8")
            # [len_first: 2B][first_bytes][len_second: 2B][second_bytes]
            f.write(struct.pack(">H", len(fb)))
            f.write(fb)
            f.write(struct.pack(">H", len(sb)))
            f.write(sb)


def read_vocab_dict(path):
    vocab_to_id = {}
    id_to_vocab = {}
    bpe_ranks = {}
    with open(path, 'rb') as f:
        (total_tokens,) = struct.unpack('>I', f.read(4))
        for _ in range(total_tokens):
            token_id, token_len = struct.unpack('>IH', f.read(6))
            token_bytes = f.read(token_len)
            token_str = token_bytes.decode('utf-8')
            vocab_to_id[token_str] = token_id
            id_to_vocab[token_id] = token_str

        (num_merges,) = struct.unpack(">I", f.read(4))
        for _ in range(num_merges):
            (fb_len,) = struct.unpack(">H", f.read(2))
            first = f.read(fb_len).decode("utf-8")
            (sb_len,) = struct.unpack(">H", f.read(2))
            second = f.read(sb_len).decode("utf-8")
            bpe_ranks[(first, second)] = _
    return vocab_to_id, id_to_vocab, bpe_ranks


def save_tokenizer(vocab_to_id, bpe_ranks, path):
    data = {
        "vocab": vocab_to_id,
        # Serialize list of merges in priority order: [["first", "second"], ...]
        "merges": sorted(bpe_ranks.keys(), key=lambda pair: bpe_ranks[pair]),
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False)


def load_tokenizer(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    vocab_to_id = data["vocab"]
    id_to_vocab = {v: k for k, v in vocab_to_id.items()}

    # Rebuild bpe_ranks mapping (first, second) -> rank_index
    bpe_ranks = {tuple(pair): idx for idx, pair in enumerate(data["merges"])}

    return vocab_to_id, id_to_vocab, bpe_ranks


# def read_and_decode_vocab(path):
#     bytes_to_id, _ = read_vocab_dict(path)
#     word_to_id = {}
#     for token_bytes, idx in bytes_to_id.items():
#         # 1. Decode bytes to the unicode token string used during training
#         token_str = token_bytes.decode("utf-8")
#         # 2. Invert the byte_encoder mapping to restore original text
#         raw_bytes = bytes([byte_decoder[ch] for ch in token_str if ch in byte_decoder])
#         word_to_id[raw_bytes.decode("utf-8", errors="replace")] = idx
#     return word_to_id

# vocab_to_id, id_to_vocab, bpe_ranks = train_bpe_fast("output.txt", target_vocab_size=128000, num_workers=64)
# save_vocab_dict(vocab_to_id,  bpe_ranks, "vocab_id.vocab")
# save_tokenizer(vocab_to_id, bpe_ranks, "vocab_id_other.vocab")
vocab_to_id, id_to_vocab, bpe_ranks = read_vocab_dict("../vocab_id.vocab")

# test_text = "The Empire will defend the war!"
# encoded_ids = encode(test_text, vocab_to_id, bpe_ranks)
# decoded_text = decode(encoded_ids, id_to_vocab)
#
# print(f"\nOriginal text : {test_text}")
# print(f"Token IDs     : {encoded_ids}")
# print(f"Decoded text  : {decoded_text}")
# print(f"Round-trip OK : {test_text == decoded_text}")

# word_to_id = {byte.decode('utf-8', errors='ignore') : idx for byte, idx in bytes_to_id.items()}
# print(vocab_to_id)
