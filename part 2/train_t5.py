import os
import argparse
from tqdm import tqdm

import torch
from transformers import T5TokenizerFast

from load_data import load_t5_data
from t5_utils import (
    initialize_model,
    initialize_optimizer_and_scheduler,
    save_model,
    load_model_from_checkpoint,
    setup_wandb,
)
from utils import compute_metrics, save_queries_and_records


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MAX_GEN_LEN = 256   # allow full SQL generation
NUM_BEAMS   = 5     # stable and strong beam search


# --------------------------------------------------------------------
# Argument parser
# --------------------------------------------------------------------
def get_args():
    parser = argparse.ArgumentParser(description="T5 Training Script")

    parser.add_argument("--finetune", action="store_true")

    # Optimization hyperparameters
    parser.add_argument("--optimizer_type", type=str, default="AdamW")
    parser.add_argument("--learning_rate", type=float, default=5e-4)
    parser.add_argument("--weight_decay", type=float, default=0.01)

    parser.add_argument("--scheduler_type", type=str, default="linear",
                        choices=["none", "cosine", "linear"])
    parser.add_argument("--num_warmup_epochs", type=int, default=1)
    parser.add_argument("--max_n_epochs", type=int, default=20)
    parser.add_argument("--patience_epochs", type=int, default=3)

    parser.add_argument("--use_wandb", action="store_true")
    parser.add_argument("--experiment_name", type=str, default="t5_ft_run")

    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--test_batch_size", type=int, default=16)

    return parser.parse_args()


# --------------------------------------------------------------------
# Training epoch
# --------------------------------------------------------------------
def train_epoch(args, model, train_loader, optimizer, scheduler):
    model.train()
    total_loss = 0.0
    total_steps = 0

    for enc, mask, dec_in, dec_tgt, _ in tqdm(train_loader):
        enc = enc.to(DEVICE)
        mask = mask.to(DEVICE)
        dec_in = dec_in.to(DEVICE)
        dec_tgt = dec_tgt.to(DEVICE)

        optimizer.zero_grad()

        outputs = model(
            input_ids=enc,
            attention_mask=mask,
            decoder_input_ids=dec_in,
            labels=dec_tgt,   # T5 handles shift internally
        )
        loss = outputs.loss

        loss.backward()
        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        total_loss += loss.item()
        total_steps += 1

    return total_loss / max(total_steps, 1)


# --------------------------------------------------------------------
# Evaluation epoch
# --------------------------------------------------------------------
def eval_epoch(args, model, dev_loader, gt_sql_path, model_sql_path,
               gt_record_path, model_record_path):
    tokenizer = T5TokenizerFast.from_pretrained("google-t5/t5-small")

    model.eval()
    total_loss = 0.0
    total_steps = 0
    all_sql_queries = []

    with torch.no_grad():
        for enc, mask, dec_in, dec_tgt, _ in tqdm(dev_loader):
            enc = enc.to(DEVICE)
            mask = mask.to(DEVICE)
            dec_in = dec_in.to(DEVICE)
            dec_tgt = dec_tgt.to(DEVICE)

            # ----- Loss -----
            outputs = model(
                input_ids=enc,
                attention_mask=mask,
                decoder_input_ids=dec_in,
                labels=dec_tgt
            )
            loss = outputs.loss
            total_loss += loss.item()
            total_steps += 1

            # ----- Generation -----
            gen_ids = model.generate(
                input_ids=enc,
                attention_mask=mask,
                max_length=MAX_GEN_LEN,
                num_beams=NUM_BEAMS,
                early_stopping=True,
                decoder_start_token_id=tokenizer.pad_token_id,
            )
            decoded_sql = tokenizer.batch_decode(gen_ids, skip_special_tokens=True)
            all_sql_queries.extend(decoded_sql)

    avg_loss = total_loss / max(total_steps, 1)

    # Save predictions & compute metrics
    save_queries_and_records(all_sql_queries, model_sql_path, model_record_path)

    sql_em, record_em, record_f1, model_errs = compute_metrics(
        gt_sql_path, model_sql_path, gt_record_path, model_record_path
    )
    error_rate = len(model_errs) / len(all_sql_queries)

    return avg_loss, record_f1, record_em, sql_em, error_rate


# --------------------------------------------------------------------
# Main training loop
# --------------------------------------------------------------------
def train(args, model, train_loader, dev_loader, optimizer, scheduler):
    best_f1 = -1
    epochs_without_improvement = 0

    model_type = "ft" if args.finetune else "scr"
    checkpoint_dir = os.path.join("checkpoints", f"{model_type}_experiments", args.experiment_name)
    os.makedirs(checkpoint_dir, exist_ok=True)
    args.checkpoint_dir = checkpoint_dir

    gt_sql_path = "data/dev.sql"
    gt_record_path = "records/ground_truth_dev.pkl"
    model_sql_path = f"results/{args.experiment_name}_dev.sql"
    model_record_path = f"records/{args.experiment_name}_dev.pkl"

    for epoch in range(args.max_n_epochs):
        train_loss = train_epoch(args, model, train_loader, optimizer, scheduler)
        print(f"[Epoch {epoch}] Train loss = {train_loss:.6f}")

        dev_loss, rec_f1, rec_em, sql_em, err_rate = eval_epoch(
            args, model, dev_loader,
            gt_sql_path, model_sql_path,
            gt_record_path, model_record_path
        )
        print(f"[Epoch {epoch}] Dev loss = {dev_loss:.6f} | F1 = {rec_f1:.6f} | EM = {rec_em:.6f} | SQL-EM = {sql_em:.6f}")
        print(f"[Epoch {epoch}] SQL error rate = {err_rate*100:.2f}%")

        if args.use_wandb:
            wandb.log({
                "train/loss": train_loss,
                "dev/loss": dev_loss,
                "dev/record_f1": rec_f1,
                "dev/record_em": rec_em,
                "dev/sql_em": sql_em,
                "dev/error_rate": err_rate,
            })

        # Track best model
        if rec_f1 > best_f1:
            best_f1 = rec_f1
            epochs_without_improvement = 0
            save_model(checkpoint_dir, model, best=True)
        else:
            epochs_without_improvement += 1

        save_model(checkpoint_dir, model, best=False)

        if epochs_without_improvement >= args.patience_epochs:
            print("Early stopping triggered.")
            break


# --------------------------------------------------------------------
# Test inference
# --------------------------------------------------------------------
def test_inference(args, model, test_loader, model_sql_path, model_record_path):
    tokenizer = T5TokenizerFast.from_pretrained("google-t5/t5-small")

    model.eval()
    all_sql_queries = []

    with torch.no_grad():
        for enc, mask, initial_in in tqdm(test_loader):
            enc = enc.to(DEVICE)
            mask = mask.to(DEVICE)

            gen_ids = model.generate(
                input_ids=enc,
                attention_mask=mask,
                max_length=MAX_GEN_LEN,
                num_beams=NUM_BEAMS,
                early_stopping=True,
                decoder_start_token_id=tokenizer.pad_token_id,
            )
            decoded = tokenizer.batch_decode(gen_ids, skip_special_tokens=True)
            all_sql_queries.extend(decoded)

    save_queries_and_records(all_sql_queries, model_sql_path, model_record_path)


# --------------------------------------------------------------------
# Script entry point
# --------------------------------------------------------------------
def main():
    args = get_args()

    if args.use_wandb:
        setup_wandb(args)

    train_loader, dev_loader, test_loader = load_t5_data(
        args.batch_size, args.test_batch_size
    )
    model = initialize_model(args)
    optimizer, scheduler = initialize_optimizer_and_scheduler(
        args, model, len(train_loader)
    )

    train(args, model, train_loader, dev_loader, optimizer, scheduler)

    # Evaluate using the best checkpoint
    model = load_model_from_checkpoint(args, best=True)
    model.eval()

    dev_sql_path = f"results/{args.experiment_name}_dev_best.sql"
    dev_record_path = f"records/{args.experiment_name}_dev_best.pkl"

    test_sql_path = f"results/{args.experiment_name}_test.sql"
    test_record_path = f"records/{args.experiment_name}_test.pkl"

    test_inference(args, model, test_loader, test_sql_path, test_record_path)


if __name__ == "__main__":
    main()
