import datasets
from datasets import load_dataset
from transformers import AutoTokenizer
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification
from torch.optim import AdamW
from transformers import get_scheduler
import torch
from tqdm.auto import tqdm
import evaluate
import random
import argparse
import re
from nltk.corpus import wordnet
from nltk import word_tokenize
from nltk.tokenize.treebank import TreebankWordDetokenizer

random.seed(0)


def example_transform(example):
    example["text"] = example["text"].lower()
    return example

def simple_tokenize(text: str):
    """
    A very simple tokenizer that splits text into words and punctuation.
    This does NOT rely on NLTK's punkt models.
    """
    return re.findall(r"\w+|[^\w\s]", text, re.UNICODE)

# Keyboard adjacency map for introducing realistic typos
KEYBOARD_NEIGHBORS = {
    "q": ["w", "a"],
    "w": ["q", "e", "a", "s"],
    "e": ["w", "r", "s", "d"],
    "r": ["e", "t", "d", "f"],
    "t": ["r", "y", "f", "g"],
    "y": ["t", "u", "g", "h"],
    "u": ["y", "i", "h", "j"],
    "i": ["u", "o", "j", "k"],
    "o": ["i", "p", "k", "l"],
    "p": ["o", "l"],
    "a": ["q", "w", "s", "z"],
    "s": ["a", "w", "e", "d", "z", "x"],
    "d": ["s", "e", "r", "f", "x", "c"],
    "f": ["d", "r", "t", "g", "c", "v"],
    "g": ["f", "t", "y", "h", "v", "b"],
    "h": ["g", "y", "u", "j", "b", "n"],
    "j": ["h", "u", "i", "k", "n", "m"],
    "k": ["j", "i", "o", "l", "m"],
    "l": ["k", "o", "p"],
    "z": ["a", "s", "x"],
    "x": ["z", "s", "d", "c"],
    "c": ["x", "d", "f", "v"],
    "v": ["c", "f", "g", "b"],
    "b": ["v", "g", "h", "n"],
    "n": ["b", "h", "j", "m"],
    "m": ["n", "j", "k"],
}


def apply_typo(token: str) -> str:
    """
    Introduce a single-character keyboard-neighbor typo into `token`.
    Only operates on alphabetic characters and keeps non-letters intact.
    """
    if not token.isalpha() or len(token) < 3:
        return token

    chars = list(token)
    candidate_positions = [i for i, ch in enumerate(chars) if ch.isalpha()]
    if not candidate_positions:
        return token

    pos = random.choice(candidate_positions)
    ch = chars[pos]
    lower_ch = ch.lower()

    if lower_ch not in KEYBOARD_NEIGHBORS:
        return token

    neighbor = random.choice(KEYBOARD_NEIGHBORS[lower_ch])

    # Preserve original casing
    if ch.isupper():
        neighbor = neighbor.upper()

    chars[pos] = neighbor
    return "".join(chars)

### Rough guidelines --- typos
# For typos, you can try to simulate nearest keys on the QWERTY keyboard for some of the letter (e.g. vowels)
# You can randomly select each word with some fixed probability, and replace random letters in that word with one of the
# nearest keys on the keyboard. You can vary the random probablity or which letters to use to achieve the desired accuracy.


### Rough guidelines --- synonym replacement
# For synonyms, use can rely on wordnet (already imported here). Wordnet (https://www.nltk.org/howto/wordnet.html) includes
# something called synsets (which stands for synonymous words) and for each of them, lemmas() should give you a possible synonym word.
# You can randomly select each word with some fixed probability to replace by a synonym.


def custom_transform(example):
    ################################
    ##### YOUR CODE BEGINGS HERE ###

    # Design and implement the transformation as mentioned in pdf
    # You are free to implement any transformation but the comments at the top roughly describe
    # how you could implement two of them --- synonym replacement and typos.

    # You should update example["text"] using your transformation
    """
    Custom transformation used for Q2.

    We apply keyboard-neighbor typos to a subset of alphabetic tokens.
    The overall semantics and sentiment label should remain the same for humans,
    but the model sees perturbed surface forms.
    """
    text = example["text"]
    tokens = simple_tokenize(text)
    detok = TreebankWordDetokenizer()

    # Make the corruption a bit stronger so that accuracy drops > 4 points
    p_typo = 0.45  # was 0.25

    new_tokens = []
    for tok in tokens:
        # lower the length threshold from 4 to 3 so that more words can be noised
        if tok.isalpha() and len(tok) >= 3 and random.random() < p_typo:
            new_tokens.append(apply_typo(tok))
        else:
            new_tokens.append(tok)

    example["text"] = detok.detokenize(new_tokens)
    return example
