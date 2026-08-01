from collections import Counter, defaultdict
import heapq
from multiprocessing import Pool
import os
import pickle
import regex
import resource
import sys
import time
import tracemalloc

from cs336_basics.pretokenization_example import find_chunk_boundaries


SPLIT_SPECIAL_TOKEN = "<|endoftext|>"
SPLIT_SPECIAL_TOKEN_BYTES = SPLIT_SPECIAL_TOKEN.encode("utf-8")
SPECIAL_TOKENS = [SPLIT_SPECIAL_TOKEN]
PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

NUM_PROCESSES = 4
# Keep task size smaller than each worker's full share of the corpus.
NUM_CHUNKS = 100
# Rebuild the pair heap once it grows past this multiple of the live pair count.
# The heap uses lazy deletion, so stale entries accumulate and are never swept otherwise.
HEAP_REBUILD_FACTOR = 4

# --- diagnostics: temporary instrumentation, strip once the numbers are understood ---
DIAG = True
# ru_maxrss is bytes on macOS, KiB on Linux
_RSS_SCALE = 1 if sys.platform == "darwin" else 1024
_T0 = time.monotonic()


def _gb(n: float) -> str:
  return f"{n / 2**30:7.3f}GB"


def _peak_rss(who: int = resource.RUSAGE_SELF) -> int:
  return resource.getrusage(who).ru_maxrss * _RSS_SCALE


def report(label: str, children: bool = False) -> None:
  if not DIAG:
    return
  parts = [f"[{time.monotonic() - _T0:7.1f}s] {label:<24}", f"self_peak={_gb(_peak_rss())}"]
  if children:
    parts.append(f"child_peak={_gb(_peak_rss(resource.RUSAGE_CHILDREN))}")
  if tracemalloc.is_tracing():
    cur, peak = tracemalloc.get_traced_memory()
    parts.append(f"py_cur={_gb(cur)} py_peak={_gb(peak)}")
  print("  ".join(parts), flush=True)


def report_structures(label, word_counts, token_counts, pairs_to_words, words_to_tokens, pair_heap) -> None:
  if not DIAG:
    return
  members = sum(len(s) for s in pairs_to_words.values())
  empties = sum(1 for s in pairs_to_words.values() if not s)
  held = sum(len(v) for v in words_to_tokens.values())
  print(
    f"{'':11}{label:<24}  words={len(word_counts):>9,}  pairs={len(token_counts):>9,}"
    f"  p2w_keys={len(pairs_to_words):>9,}  p2w_members={members:>11,}  p2w_empty={empties:>9,}"
    f"  w2t_tokens={held:>11,}  heap={len(pair_heap):>11,}",
    flush=True,
  )

# print(regex.findall(PAT, "some text that i'll pre-tokenize"))


def display_token(token: bytes) -> str:
  return token.decode("utf-8", errors="replace")


class ReversePair:
  __slots__ = ("pair",)

  def __init__(self, pair: tuple[bytes, bytes]):
    self.pair = pair

  def __lt__(self, other):
    return self.pair > other.pair


def push_pair(pair_heap, token_counts, pair):
  count = token_counts.get(pair, 0)
  if count > 0:
    heapq.heappush(pair_heap, (-count, ReversePair(pair), pair))


def maybe_rebuild_heap(pair_heap, token_counts):
  """Drop stale lazy-deletion entries by rebuilding from the authoritative counts."""
  if len(pair_heap) <= HEAP_REBUILD_FACTOR * max(len(token_counts), 1):
    return pair_heap
  rebuilt = [(-count, ReversePair(pair), pair) for pair, count in token_counts.items() if count > 0]
  heapq.heapify(rebuilt)
  if DIAG:
    print(f"{'':11}heap rebuild: {len(pair_heap):>11,} -> {len(rebuilt):>11,}", flush=True)
  return rebuilt


def pop_best_pair(pair_heap, token_counts):
  while pair_heap:
    neg_count, _, pair = heapq.heappop(pair_heap)
    if token_counts.get(pair, 0) == -neg_count:
      return pair
  raise ValueError("No pairs left to merge")



corpus = """low low low low low
lower lower widest widest widest
newest newest newest newest newest newest"""

# Pre tokenize 

def pretokenize_chunk(input_path: str, start: int, end: int) -> Counter:
  with open(input_path, "rb") as f:
    f.seek(start)
    chunk = f.read(end - start).decode("utf-8", errors="ignore")

  delimiters = "|".join(regex.escape(st) for st in SPECIAL_TOKENS)

  counts = Counter()
  docs = regex.split(delimiters, chunk)
  for doc in docs:
    for i in regex.finditer(PAT, doc):
      counts[i.group().encode("utf-8")] += 1

  return counts


def pretokenize_chunk_from_task(task: tuple[str, int, int]) -> Counter:
  return pretokenize_chunk(*task)


def pretokenize(input_path: str, num_processes: int, num_chunks: int = NUM_CHUNKS):
  with open(input_path, "rb") as f:
      boundaries = find_chunk_boundaries(f, num_chunks, SPLIT_SPECIAL_TOKEN_BYTES)

      tasks = [(input_path, start, end) for start, end in zip(boundaries[:-1], boundaries[1: ])]

      if DIAG:
        sizes = [end - start for _, start, end in tasks]
        print(
          f"{'':11}chunking: {len(tasks)} chunk(s) requested={num_chunks} for {num_processes} process(es)"
          f"  min={min(sizes):,}  max={max(sizes):,}  mean={sum(sizes) // len(sizes):,} bytes",
          flush=True,
        )

      agg_counter = Counter()
      done = 0
      with Pool(processes=num_processes) as pool:
        for c in pool.imap_unordered(pretokenize_chunk_from_task, tasks):
          agg_counter += c
          done += 1
          if DIAG and (done % 10 == 0 or done == len(tasks)):
            report(f"merged {done}/{len(tasks)} ({len(agg_counter):,} uniq)")

  return agg_counter

# Tokenize

def tokenize(counter: Counter[bytes]): 
    pairs_to_words = defaultdict(set)
    words_to_tokens = defaultdict(list)
    token_counter = Counter()
    for word in counter:
      word_tokens = [bytes([b]) for b in word]
      words_to_tokens[word] = word_tokens
      for pair in zip(word_tokens, word_tokens[1: ]): 
        pairs_to_words[pair].add(word)
        token_counter[pair] += counter[word]

    return token_counter, pairs_to_words, words_to_tokens


pretokenized = [j for i in corpus.splitlines() for j in i.strip().split(" ")]


base_table = Counter()
for el in pretokenized: 
  t = tuple(bytes([b]) for b in el.encode("utf-8"))
  base_table[t] += 1


def aggregate(table): 
  freqs = Counter()
  for tup in table: 
    for pair in zip(tup, tup[1: ]): 
      freqs[pair] += table[tup]
  
  return freqs

def merge_table(table, val): 
  new_table = {}
  for tup, result in table.items(): 
    buf = []
    i = 0
    n = len(tup)
    while i < n: 
      if i < len(tup) - 1 and tup[i] + tup[i + 1] == val:  
        buf.append(val)
        i += 2
      else: 
        buf.append(tup[i])
        i += 1
    new_table[tuple(buf)] = result 
  return new_table


def train_tokenizer_step(table): 
  freqs = aggregate(table)
  m_pair = max(freqs, key=lambda x: (freqs[x], x))
  new_table = merge_table(table, m_pair[0] + m_pair[1])
  return new_table, m_pair

def train_tokenizer(table, num_merges): 
  vocab = [bytes([i]) for i in range(256)] + [b"<|endoftext|>"]
  merge_list = []
  for _ in range(num_merges):
    table, merge = train_tokenizer_step(table)
    merge_list.append(merge)

  
  for m in merge_list: 
    vocab.append(m[0] + m[1])

  return table, vocab, merge_list


def update_tables(
    word_counts: Counter[bytes],
    token_counts: Counter[tuple[bytes, bytes]],
    pairs_to_words: defaultdict[tuple[bytes, bytes], set[bytes]],
    words_to_tokens: defaultdict[bytes, list[bytes]],
    pair_heap: list[tuple[int, ReversePair, tuple[bytes, bytes]]],
    merge: tuple[bytes, bytes],
):
  post_merge_token = merge[0] + merge[1]
  changed_pairs = set()

  def remove_stale_entries(pair, word): 
    pairs_to_words[pair].discard(word)
    token_counts[pair] -= word_counts[word]
    changed_pairs.add(pair)
    if token_counts[pair] <= 0:
      del token_counts[pair]

  words_to_be_updated = list(pairs_to_words[merge])
  for word in words_to_be_updated:
    w = words_to_tokens[word]
    for w_pair in zip(w, w[1:]):
      remove_stale_entries(w_pair, word)

    buf = []
    i = 0 
    w_len = len(w)
    while i < w_len: 
      if i < w_len - 1 and (w[i], w[i + 1]) == merge:
        buf.append(post_merge_token)
        i += 2
      else: 
        buf.append(w[i])
        i += 1

    words_to_tokens[word] = buf
    for w_pair in zip(buf, buf[1:]):
      pairs_to_words[w_pair].add(word)
      token_counts[w_pair] += word_counts[word]
      changed_pairs.add(w_pair)

  for pair in changed_pairs:
    push_pair(pair_heap, token_counts, pair)

  return token_counts, pairs_to_words, words_to_tokens, pair_heap

    
    






def train_bpe(
    input_path: str,
    vocab_size: int,
    special_tokens: list[str],
    report_every: int = 100,
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
  num_merges = vocab_size - len(special_tokens) - 256
  vocabulary = [bytes([i]) for i in range(256)] + [st.encode("utf-8") for st in special_tokens]
  merges = []
  report("start")
  word_counts = pretokenize(input_path=input_path, num_processes=NUM_PROCESSES)
  report("after pretokenize", children=True)
  token_counts, pairs_to_words, words_to_tokens = tokenize(word_counts)
  report("after tokenize")
  pair_heap = []
  for pair in token_counts:
    push_pair(pair_heap, token_counts, pair)
  report("after heap build")
  report_structures("initial", word_counts, token_counts, pairs_to_words, words_to_tokens, pair_heap)

  for n in range(num_merges):
    merge = pop_best_pair(pair_heap, token_counts)
    post_merge_token = merge[0] + merge[1]
    merges.append(merge)
    vocabulary.append(post_merge_token)
    update_result = update_tables(word_counts, token_counts, pairs_to_words, words_to_tokens, pair_heap, merge)
    token_counts, pairs_to_words, words_to_tokens, pair_heap = update_result
    pair_heap = maybe_rebuild_heap(pair_heap, token_counts)

    if DIAG and report_every and (n + 1) % report_every == 0:
      report(f"merge {n + 1}/{num_merges}")
      report_structures(
        f"merge {n + 1}", word_counts, token_counts, pairs_to_words, words_to_tokens, pair_heap
      )

  report("done")
  return dict(enumerate(vocabulary)), merges

    



  


"""
Steps: 
    1. Get max
    2. merge and record it 
    3. delete that pair from token counts
    4. grab all words where that pair showed up, do old tokenization, subtract those freqs from token counter, do new tokenization, add those freqs back?  
"""

if __name__ == "__main__":
  input_path = os.environ.get("BPE_INPUT", "data/owt_train.txt")
  vocab_size = int(os.environ.get("BPE_VOCAB", "32000"))
  report_every = int(os.environ.get("BPE_REPORT_EVERY", "100"))
  if os.environ.get("BPE_TRACEMALLOC") == "1":
    tracemalloc.start()

  file_bytes = os.path.getsize(input_path)
  print(
    f"input={input_path} ({file_bytes:,} bytes)  vocab_size={vocab_size}  num_processes={NUM_PROCESSES}"
    f"  tracemalloc={tracemalloc.is_tracing()}",
    flush=True,
  )

  vocab, merges = train_bpe(
    input_path, vocab_size=vocab_size, special_tokens=[SPLIT_SPECIAL_TOKEN], report_every=report_every
  )
  vocab_output_path = f"src/bpe/owt_vocab{vocab_size}.pkl"
  merges_output_path = f"src/bpe/owt_merges{vocab_size}.pkl"
  with open(vocab_output_path, "wb") as f:
    pickle.dump(vocab, f)
  with open(merges_output_path, "wb") as f:
    pickle.dump(merges, f)

  # print(f"\nwrote tokenizer to: {output_path}")
  # print(f"\nvocab size: {len(vocab)}")
  # print(f"num merges: {len(merges)}")
  # print("\nmerges:")
  # for idx, merge in enumerate(merges, start=1):
  #   print(f"{idx:03d}: {display_token(merge[0])!r} + {display_token(merge[1])!r} -> {display_token(merge[0] + merge[1])!r}")
