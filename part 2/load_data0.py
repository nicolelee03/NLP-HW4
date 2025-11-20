import os
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

import nltk
nltk.download('punkt')

from transformers import T5TokenizerFast
import torch

# We keep PAD_IDX = 0 to stay consistent with the rest of the codebase
PAD_IDX = 0

# Task prefix used for the encoder side
TASK_PREFIX = "translate to sql: "
MAX_SRC_LEN = 128
MAX_TGT_LEN = 256

# Create a single tokenizer instance to be reused
TOKENIZER = T5TokenizerFast.from_pretrained("google-t5/t5-small")
PAD_ID = TOKENIZER.pad_token_id   # should be 0 for T5
EOS_ID = TOKENIZER.eos_token_id


def load_lines(path):
    with open(path, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f]
    return lines


class T5Dataset(Dataset):
    """
    Simple dataset that keeps raw NL and SQL strings.
    Tokenization is done inside the collate functions.
    """

    def __init__(self, data_folder, split):
        """
        Args:
            data_folder (str): path to the data directory.
            split (str): "train", "dev", or "test".
        """
        assert split in {"train", "dev", "test"}
        self.split = split

        nl_path = os.path.join(data_folder, f"{split}.nl")
        self.nl = load_lines(nl_path)

        if split != "test":
            sql_path = os.path.join(data_folder, f"{split}.sql")
            self.sql = load_lines(sql_path)
            assert len(self.nl) == len(self.sql), "NL and SQL line count mismatch"
        else:
            self.sql = None

    def __len__(self):
        return len(self.nl)

    def __getitem__(self, idx):
        item = {"nl": self.nl[idx]}
        if self.sql is not None:
            item["sql"] = self.sql[idx]
        return item

    # The skeleton had process_data, but we no longer need it, so keep a stub:
    def process_data(self, data_folder, split, tokenizer):
        return

def _preprocess_nl(text: str) -> str:
    """
    Basic preprocessing for the NL side: strip + lowercase + prefix.
    """
    text = text.strip().lower()
    return TASK_PREFIX + text


def _preprocess_sql(text: str) -> str:
    """
    Basic preprocessing for the SQL side: strip + lowercase.
    We do not add special tokens here; they are handled when building labels.
    """
    text = text.strip().lower()
    return text


def normal_collate_fn(batch):
    """
    Collation function to perform dynamic padding for training and evaluation
    with the development or validation set.

    Inputs:
        * batch: list of dicts with keys "nl" and "sql".

    Returns:
        * encoder_ids:  (B, Tenc) LongTensor to be fed into the T5 encoder.
        * encoder_mask: (B, Tenc) LongTensor mask (1 for non-padding, 0 for padding).
        * decoder_inputs:  (B, Tdec) LongTensor to be fed into the T5 decoder.
        * decoder_targets: (B, Tdec) LongTensor of target tokens (the "next token" for each position).
        * initial_decoder_inputs: (B, 1) LongTensor, the first decoder input token
          (only used in evaluation if needed).
    """
    enc_ids_list = []
    enc_mask_list = []
    dec_in_list = []
    dec_tgt_list = []

    for item in batch:
        # ---------- Encoder side ----------
        src_text = _preprocess_nl(item["nl"])
        enc = TOKENIZER(
            src_text,
            max_length=MAX_SRC_LEN,
            truncation=True,
            return_attention_mask=True,
        )
        enc_ids = torch.tensor(enc["input_ids"], dtype=torch.long)
        enc_mask = torch.tensor(enc["attention_mask"], dtype=torch.long)
        enc_ids_list.append(enc_ids)
        enc_mask_list.append(enc_mask)

        # ---------- Decoder side ----------
        tgt_text = _preprocess_sql(item["sql"])
        tgt_ids = TOKENIZER(
            tgt_text,
            max_length=MAX_TGT_LEN - 1,  # leave space for EOS
            truncation=True,
            add_special_tokens=False,
        )["input_ids"]

        # decoder_inputs: [PAD] + tgt_ids
        dec_in = [PAD_ID] + tgt_ids
        # decoder_targets: tgt_ids + [EOS]
        dec_tgt = tgt_ids + [EOS_ID]

        dec_in_list.append(torch.tensor(dec_in, dtype=torch.long))
        dec_tgt_list.append(torch.tensor(dec_tgt, dtype=torch.long))

    # Pad all sequences
    encoder_ids = pad_sequence(enc_ids_list, batch_first=True, padding_value=PAD_ID)
    encoder_mask = pad_sequence(enc_mask_list, batch_first=True, padding_value=0)

    decoder_inputs = pad_sequence(dec_in_list, batch_first=True, padding_value=PAD_ID)
    decoder_targets = pad_sequence(dec_tgt_list, batch_first=True, padding_value=PAD_ID)

    # Initial decoder inputs: a single PAD token for each example
    batch_size = encoder_ids.size(0)
    initial_decoder_inputs = torch.full((batch_size, 1), PAD_ID, dtype=torch.long)

    return (
        encoder_ids,
        encoder_mask,
        decoder_inputs,
        decoder_targets,
        initial_decoder_inputs,
    )


def test_collate_fn(batch):
    """
    Collation function to perform dynamic padding for inference on the test set.

    Inputs:
        * batch: list of dicts with key "nl".

    Returns:
        * encoder_ids: (B, T) LongTensor to be fed into the T5 encoder.
        * encoder_mask: (B, T) LongTensor mask.
        * initial_decoder_inputs: (B, 1) LongTensor, first decoder input token.
    """
    enc_ids_list = []
    enc_mask_list = []

    for item in batch:
        src_text = _preprocess_nl(item["nl"])
        enc = TOKENIZER(
            src_text,
            max_length=MAX_SRC_LEN,
            truncation=True,
            return_attention_mask=True,
        )
        enc_ids = torch.tensor(enc["input_ids"], dtype=torch.long)
        enc_mask = torch.tensor(enc["attention_mask"], dtype=torch.long)
        enc_ids_list.append(enc_ids)
        enc_mask_list.append(enc_mask)

    encoder_ids = pad_sequence(enc_ids_list, batch_first=True, padding_value=PAD_ID)
    encoder_mask = pad_sequence(enc_mask_list, batch_first=True, padding_value=0)

    batch_size = encoder_ids.size(0)
    initial_decoder_inputs = torch.full((batch_size, 1), PAD_ID, dtype=torch.long)

    return encoder_ids, encoder_mask, initial_decoder_inputs


def get_dataloader(batch_size, split):
    data_folder = "data"
    dset = T5Dataset(data_folder, split)
    shuffle = split == "train"
    collate_fn = normal_collate_fn if split != "test" else test_collate_fn

    dataloader = DataLoader(
        dset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate_fn,
    )
    return dataloader


def load_t5_data(batch_size, test_batch_size):
    train_loader = get_dataloader(batch_size, "train")
    dev_loader = get_dataloader(test_batch_size, "dev")
    test_loader = get_dataloader(test_batch_size, "test")
    return train_loader, dev_loader, test_loader


def load_prompting_data(data_folder):
    """
    Helper for prompting-based experiments.
    """
    train_x = load_lines(os.path.join(data_folder, "train.nl"))
    train_y = load_lines(os.path.join(data_folder, "train.sql"))
    dev_x = load_lines(os.path.join(data_folder, "dev.nl"))
    dev_y = load_lines(os.path.join(data_folder, "dev.sql"))
    test_x = load_lines(os.path.join(data_folder, "test.nl"))
    return train_x, train_y, dev_x, dev_y, test_x