import os
import re
from tqdm import tqdm

import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

from transformers import T5TokenizerFast

# --------------------------------------------------------------------
# Global tokenizer for the entire module
# --------------------------------------------------------------------
TOKENIZER = T5TokenizerFast.from_pretrained("google-t5/t5-small")
PAD_ID = TOKENIZER.pad_token_id
EOS_ID = TOKENIZER.eos_token_id

MAX_SRC_LEN = 128      # encoder input length
MAX_TGT_LEN = 256      # decoder (SQL) length

PREFIX = "translate English to SQL: "


# --------------------------------------------------------------------
# Helper: read one line per sample
# --------------------------------------------------------------------
def _read_lines(path):
    with open(path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f]


# --------------------------------------------------------------------
# Dataset wrapper
# --------------------------------------------------------------------
class T5Dataset(Dataset):
    """
    Dataset wrapper for the text-to-SQL task.
    """

    def __init__(self, data_folder: str, split: str):
        assert split in {"train", "dev", "test"}

        nl_path = os.path.join(data_folder, f"{split}.nl")
        self.nl_lines = _read_lines(nl_path)

        if split != "test":
            sql_path = os.path.join(data_folder, f"{split}.sql")
            self.sql_lines = _read_lines(sql_path)
            assert len(self.sql_lines) == len(self.nl_lines)
        else:
            self.sql_lines = None

        self.split = split

    def __len__(self):
        return len(self.nl_lines)

    def __getitem__(self, idx):
        nl = self.nl_lines[idx]

        if self.split == "test":
            return nl
        else:
            sql = self.sql_lines[idx]
            return nl, sql


# --------------------------------------------------------------------
# Preprocessing functions
# --------------------------------------------------------------------
def preprocess_nl(text: str) -> str:
    text = text.strip()
    return PREFIX + text


def preprocess_sql(text: str) -> str:
    text = text.strip()
    text = text.replace("\n", " ")
    text = re.sub(r"\s+", " ", text)
    return text


# --------------------------------------------------------------------
# Collate functions
# --------------------------------------------------------------------
def normal_collate_fn(batch):
    """
    Collate function for train and dev.  
    Returns:
        encoder_ids         B x T_enc
        encoder_mask        B x T_enc
        decoder_inputs      B x T_dec     ([PAD] + SQL_tokens)
        decoder_targets     B x T_dec     (SQL_tokens + [EOS])
        initial_decoder     B x 1         (always PAD)
    """
    encoder_id_list = []
    encoder_mask_list = []
    decoder_in_list = []
    decoder_tgt_list = []

    for nl, sql in batch:
        # ----- Encoder side -----
        nl_proc = preprocess_nl(nl)
        enc = TOKENIZER(
            nl_proc,
            truncation=True,
            max_length=MAX_SRC_LEN,
            add_special_tokens=True,
        )
        enc_ids = torch.tensor(enc["input_ids"], dtype=torch.long)
        encoder_id_list.append(enc_ids)
        encoder_mask_list.append(torch.ones_like(enc_ids))

        # ----- Decoder side -----
        sql_proc = preprocess_sql(sql)
        dec = TOKENIZER(
            sql_proc,
            truncation=True,
            max_length=MAX_TGT_LEN - 1,
            add_special_tokens=False,   # very important!
        )
        tgt_ids = torch.tensor(dec["input_ids"], dtype=torch.long)

        # decoder_input_ids = [PAD] + tgt_ids
        dec_in = torch.cat([torch.tensor([PAD_ID]), tgt_ids], dim=0)
        # labels = tgt_ids + [EOS]
        dec_tgt = torch.cat([tgt_ids, torch.tensor([EOS_ID])], dim=0)

        decoder_in_list.append(dec_in)
        decoder_tgt_list.append(dec_tgt)

    # Pad everything
    encoder_ids = pad_sequence(encoder_id_list, batch_first=True, padding_value=PAD_ID)
    encoder_mask = (encoder_ids != PAD_ID).long()

    decoder_inputs = pad_sequence(decoder_in_list, batch_first=True, padding_value=PAD_ID)
    decoder_targets = pad_sequence(decoder_tgt_list, batch_first=True, padding_value=PAD_ID)

    initial_dec = torch.full((encoder_ids.size(0), 1), PAD_ID, dtype=torch.long)

    return encoder_ids, encoder_mask, decoder_inputs, decoder_targets, initial_dec


def test_collate_fn(batch):
    """
    Collate function for test (no SQL given).
    """
    encoder_id_list = []
    for nl in batch:
        nl_proc = preprocess_nl(nl)
        enc = TOKENIZER(
            nl_proc,
            truncation=True,
            max_length=MAX_SRC_LEN,
            add_special_tokens=True,
        )
        encoder_id_list.append(torch.tensor(enc["input_ids"], dtype=torch.long))

    encoder_ids = pad_sequence(encoder_id_list, batch_first=True, padding_value=PAD_ID)
    encoder_mask = (encoder_ids != PAD_ID).long()

    initial_dec = torch.full((encoder_ids.size(0), 1), PAD_ID, dtype=torch.long)
    return encoder_ids, encoder_mask, initial_dec


# --------------------------------------------------------------------
# Data loader constructors
# --------------------------------------------------------------------
def get_dataloader(batch_size, split):
    data_folder = "data"
    dataset = T5Dataset(data_folder, split)
    shuffle = split == "train"
    collate_fn = normal_collate_fn if split != "test" else test_collate_fn

    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn)


def load_t5_data(batch_size, test_batch_size):
    train_loader = get_dataloader(batch_size, "train")
    dev_loader   = get_dataloader(test_batch_size, "dev")
    test_loader  = get_dataloader(test_batch_size, "test")
    return train_loader, dev_loader, test_loader


# --------------------------------------------------------------------
# Prompting-based experiments (used in Q8)
# --------------------------------------------------------------------
def load_lines(path):
    return _read_lines(path)


def load_prompting_data(data_folder):
    train_x = load_lines(os.path.join(data_folder, "train.nl"))
    train_y = load_lines(os.path.join(data_folder, "train.sql"))
    dev_x   = load_lines(os.path.join(data_folder, "dev.nl"))
    dev_y   = load_lines(os.path.join(data_folder, "dev.sql"))
    test_x  = load_lines(os.path.join(data_folder, "test.nl"))
    return train_x, train_y, dev_x, dev_y, test_x
