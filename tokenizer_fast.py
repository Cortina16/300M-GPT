import struct
import re
import time

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


from functools import lru_cache

@lru_cache(maxsize=300_000)
def bpe_merge_word(word_tuple, bpe_ranks):
    if len(word_tuple) <= 1:
        return word_tuple

    tokens = list(word_tuple)
    iterations = 0
    max_iterations = 5_000

    while len(tokens) > 1:
        iterations += 1
        if iterations > max_iterations:
            break

        min_rank = float("inf")
        best_pair = None

        for i in range(len(tokens) - 1):
            pair = (tokens[i], tokens[i + 1])
            rank = bpe_ranks.get(pair, float("inf"))
            if rank < min_rank:
                min_rank = rank
                best_pair = pair

        if best_pair is None or min_rank == float("inf"):
            break

        first, second = best_pair
        new_tokens = []
        i = 0
        merged_any = False

        while i < len(tokens):
            if i < len(tokens) - 1 and tokens[i] == first and tokens[i + 1] == second:
                new_tokens.append(first + second)
                i += 2
                merged_any = True
            else:
                new_tokens.append(tokens[i])
                i += 1

        if not merged_any or len(new_tokens) >= len(tokens):
            break

        tokens = new_tokens

    return tuple(tokens)


def encode(text, vocab_to_id, bpe_ranks):
    chunks = SPLIT_PATTERN.findall(text)
    token_ids = []

    for chunk in chunks:
        raw_bytes = chunk.encode("utf-8")
        byte_tokens = tuple(byte_encoder[b] for b in raw_bytes)

        # Hits the LRU cache for repeated words
        merged_tokens = bpe_merge_word(byte_tokens, bpe_ranks)

        for token in merged_tokens:
            if token in vocab_to_id:
                token_ids.append(vocab_to_id[token])
            else:
                for b in token.encode("utf-8"):
                    char = byte_encoder.get(b)
                    if char in vocab_to_id:
                        token_ids.append(vocab_to_id[char])

    return token_ids

def read_vocab_dict(path):
    vocab_to_id = {}
    id_to_vocab = {}
    bpe_ranks = {}
    with open(path, "rb") as f:
        (total_tokens,) = struct.unpack(">I", f.read(4))
        for _ in range(total_tokens):
            token_id, token_len = struct.unpack(">IH", f.read(6))
            token_bytes = f.read(token_len)
            token_str = token_bytes.decode("utf-8")
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