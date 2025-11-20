# stats_t5.py
import os
import re
from collections import Counter
from transformers import T5TokenizerFast

DATA_DIR = "data"
MODEL_NAME = "google-t5/t5-small"

tokenizer = T5TokenizerFast.from_pretrained(MODEL_NAME)

MAX_SRC_LEN = 128   # for "after" stats
MAX_TGT_LEN = 256   # for "after" stats


def read_split(split):
    """Read NL and SQL for a given split ('train' or 'dev')."""
    nl_path = os.path.join(DATA_DIR, f"{split}.nl")
    sql_path = os.path.join(DATA_DIR, f"{split}.sql")

    with open(nl_path, "r", encoding="utf-8") as f:
        nl = [line.strip() for line in f]

    with open(sql_path, "r", encoding="utf-8") as f:
        sql = [line.strip() for line in f]

    if len(nl) != len(sql):
        min_len = min(len(nl), len(sql))
        print(
            f"[Warning] Mismatch between NL ({len(nl)}) and SQL ({len(sql)}) "
            f"lines in {split}; using first {min_len} pairs for statistics."
        )
        nl = nl[:min_len]
        sql = sql[:min_len]

    return nl, sql


# ----------------- preprocessing rules (for AFTER stats) ----------------- #

def preprocess_nl(text: str) -> str:
    """Preprocessing for NL side used in AFTER stats."""
    text = text.strip().lower()
    return "translate to sql: " + text


def preprocess_sql(text: str) -> str:
    """Preprocessing for SQL side used in AFTER stats."""
    text = text.strip().lower()
    # collapse whitespace
    text = text.replace("\n", " ")
    text = re.sub(r"\s+", " ", text)
    # remove trailing semicolon
    if text.endswith(";"):
        text = text[:-1].rstrip()
    return text


# ----------------- helpers to compute stats ----------------- #

def compute_token_stats(texts, preprocess=None, max_len=None):
    """
    texts: list of strings
    preprocess: function or None
    max_len: max_length for tokenizer, or None (no truncation)
    Returns: (mean_length, vocab_size, num_examples)
    """
    lengths = []
    vocab_counter = Counter()

    for t in texts:
        if preprocess is not None:
            t = preprocess(t)

        enc = tokenizer(
            t,
            truncation=(max_len is not None),
            max_length=max_len,
        )
        ids = enc["input_ids"]
        lengths.append(len(ids))
        vocab_counter.update(ids)

    mean_len = sum(lengths) / len(lengths) if lengths else 0.0
    vocab_size = len(vocab_counter)
    return mean_len, vocab_size, len(lengths)


def stats_before_after(split):
    """
    Compute BEFORE and AFTER stats for a given split.
    Returns dicts:
      before = { 'n_examples', 'mean_nl', 'mean_sql', 'vocab_nl', 'vocab_sql' }
      after  = { ... }
    """
    nl, sql = read_split(split)

    # BEFORE: only strip, no truncation, no prefix, no lowercasing
    mean_nl_before, vocab_nl_before, n_examples = compute_token_stats(
        nl, preprocess=lambda x: x.strip(), max_len=None
    )
    mean_sql_before, vocab_sql_before, _ = compute_token_stats(
        sql, preprocess=lambda x: x.strip(), max_len=None
    )

    before = {
        "n_examples": n_examples,
        "mean_nl": mean_nl_before,
        "mean_sql": mean_sql_before,
        "vocab_nl": vocab_nl_before,
        "vocab_sql": vocab_sql_before,
    }

    # AFTER: use our preprocessing + max lengths
    mean_nl_after, vocab_nl_after, n_examples_after = compute_token_stats(
        nl, preprocess=preprocess_nl, max_len=MAX_SRC_LEN
    )
    mean_sql_after, vocab_sql_after, _ = compute_token_stats(
        sql, preprocess=preprocess_sql, max_len=MAX_TGT_LEN
    )

    after = {
        "n_examples": n_examples_after,
        "mean_nl": mean_nl_after,
        "mean_sql": mean_sql_after,
        "vocab_nl": vocab_nl_after,
        "vocab_sql": vocab_sql_after,
    }

    return before, after


def print_table_stats():
    splits = ["train", "dev"]
    results_before = {}
    results_after = {}

    for sp in splits:
        before, after = stats_before_after(sp)
        results_before[sp] = before
        results_after[sp] = after

    # ----------- Print for Table 1 (BEFORE preprocessing) ----------- #
    print("=== Table 1: BEFORE preprocessing (token-level, T5 tokenizer) ===")
    print("Statistics Name           Train          Dev")
    print("-" * 55)
    print(
        f"Number of examples   {results_before['train']['n_examples']:>8d}"
        f"     {results_before['dev']['n_examples']:>8d}"
    )
    print(
        f"Mean sentence length {results_before['train']['mean_nl']:8.2f}"
        f"     {results_before['dev']['mean_nl']:8.2f}"
    )
    print(
        f"Mean SQL query length{results_before['train']['mean_sql']:8.2f}"
        f"     {results_before['dev']['mean_sql']:8.2f}"
    )
    print(
        f"Vocabulary size (NL) {results_before['train']['vocab_nl']:8d}"
        f"     {results_before['dev']['vocab_nl']:8d}"
    )
    print(
        f"Vocabulary size (SQL){results_before['train']['vocab_sql']:8d}"
        f"     {results_before['dev']['vocab_sql']:8d}"
    )
    print()

    # ----------- Print for Table 2 (AFTER preprocessing) ----------- #
    print("=== Table 2: AFTER preprocessing (token-level, T5 tokenizer) ===")
    print(f"Model name: {MODEL_NAME}")
    print("Statistics Name           Train          Dev")
    print("-" * 55)
    print(
        f"Number of examples   {results_after['train']['n_examples']:>8d}"
        f"     {results_after['dev']['n_examples']:>8d}"
    )
    print(
        f"Mean sentence length {results_after['train']['mean_nl']:8.2f}"
        f"     {results_after['dev']['mean_nl']:8.2f}"
    )
    print(
        f"Mean SQL query length{results_after['train']['mean_sql']:8.2f}"
        f"     {results_after['dev']['mean_sql']:8.2f}"
    )
    print(
        f"Vocabulary size (NL) {results_after['train']['vocab_nl']:8d}"
        f"     {results_after['dev']['vocab_nl']:8d}"
    )
    print(
        f"Vocabulary size (SQL){results_after['train']['vocab_sql']:8d}"
        f"     {results_after['dev']['vocab_sql']:8d}"
    )


if __name__ == "__main__":
    print_table_stats()
