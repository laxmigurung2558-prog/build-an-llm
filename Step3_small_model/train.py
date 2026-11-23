"""Train a tiny, randomly initialised causal transformer on fineweb_corpus.txt.

Run:  python train.py
All outputs are written to the results/ folder.
"""

import argparse
import csv
import json
import os
import re
import sys
import time

import tiktoken
import torch

from model import TinyGPT, generate_text, missing_tokens


class Tee:
    """Write everything printed to both the terminal and a log file."""

    def __init__(self, *streams):
        self.streams = streams

    def write(self, s):
        for st in self.streams:
            st.write(s)

    def flush(self):
        for st in self.streams:
            st.flush()


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--corpus", default="fineweb_corpus.txt")
    p.add_argument("--tokens", default="fineweb_tokens.txt")
    p.add_argument("--out", default="results")
    p.add_argument("--seconds", type=float, default=60.0,
                   help="training time budget (sampling time not counted)")
    p.add_argument("--prompt", default="The company is",
                   help="short sentence beginning used for every sample")
    p.add_argument("--seed", type=int, default=1234,
                   help="random seed used for every sample")
    p.add_argument("--init-seed", type=int, default=42,
                   help="seed for random weight initialisation and batches")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--block-size", type=int, default=64)
    p.add_argument("--n-layer", type=int, default=2)
    p.add_argument("--n-head", type=int, default=4)
    p.add_argument("--n-embd", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--print-every", type=int, default=5)
    p.add_argument("--sample-every", type=int, default=20)
    p.add_argument("--gen-tokens", type=int, default=40)
    p.add_argument("--temperature", type=float, default=0.8)
    return p.parse_args()


def load_and_validate_tokens(enc, corpus, tokens_path, out_dir):
    """Compare the token IDs file with a fresh o200k_base encoding of the corpus."""
    corpus_ids = enc.encode_ordinary(corpus)
    with open(tokens_path, encoding="utf-8") as f:
        file_ids = [int(x) for x in re.findall(r"\d+", f.read())]

    print(f"Corpus characters:            {len(corpus):,}")
    print(f"Token IDs in {tokens_path}: {len(file_ids):,}")
    print(f"Token IDs from re-encoding:   {len(corpus_ids):,}")

    if file_ids == corpus_ids:
        print("Validation: OK - the token IDs file matches the corpus exactly.")
        return file_ids

    first_diff = next((i for i, (a, b) in enumerate(zip(file_ids, corpus_ids))
                       if a != b), min(len(file_ids), len(corpus_ids)))
    print("Validation: MISMATCH - the token IDs file does not match the corpus "
          f"(first difference at position {first_diff}).")
    regen_path = os.path.join(out_dir, "fineweb_tokens_regenerated.txt")
    with open(regen_path, "w", encoding="utf-8") as f:
        f.write(", ".join(str(i) for i in corpus_ids))
    print(f"Regenerated the token IDs from the corpus -> {regen_path}")
    return corpus_ids


def choose_prompt(enc, stoi, prompt, token_ids):
    """Use the prompt if all its tokens are known; otherwise fall back."""
    for candidate in (prompt, " " + prompt):
        _, missing = missing_tokens(enc, stoi, candidate)
        if not missing:
            return candidate
    fallback = enc.decode(token_ids[:4])
    print(f"Note: '{prompt}' has tokens the model never saw; "
          f"using the corpus opening '{fallback}' instead.")
    return fallback


def save_plot(rows, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    steps = [r["step"] for r in rows]
    losses = [r["loss"] for r in rows]
    window = 20
    smooth = [sum(losses[max(0, i - window + 1):i + 1]) /
              len(losses[max(0, i - window + 1):i + 1])
              for i in range(len(losses))]
    plt.figure(figsize=(8, 4.5))
    plt.plot(steps, losses, alpha=0.35, label="loss per step")
    plt.plot(steps, smooth, linewidth=2, label=f"moving average ({window} steps)")
    plt.xlabel("training step")
    plt.ylabel("cross-entropy loss")
    plt.title("Tiny transformer training loss")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def main():
    args = parse_args()
    os.makedirs(args.out, exist_ok=True)
    log_file = open(os.path.join(args.out, "training_log.txt"), "w",
                    encoding="utf-8")
    sys.stdout = Tee(sys.__stdout__, log_file)

    torch.manual_seed(args.init_seed)
    enc = tiktoken.get_encoding("o200k_base")  # GPT-4o's tokenizer

    with open(args.corpus, encoding="utf-8") as f:
        corpus = f.read()
    token_ids = load_and_validate_tokens(enc, corpus, args.tokens, args.out)

    # Compact vocabulary: only the token IDs that actually occur in the corpus.
    itos = sorted(set(token_ids))           # internal id -> o200k id
    stoi = {tok: i for i, tok in enumerate(itos)}  # o200k id -> internal id
    data = torch.tensor([stoi[t] for t in token_ids], dtype=torch.long)
    print(f"Compact vocabulary: {len(itos):,} distinct tokens "
          f"(out of ~200,000 in o200k_base)")

    if len(data) <= args.block_size + 1:
        raise SystemExit("Corpus is too short for the chosen --block-size.")

    prompt = choose_prompt(enc, stoi, args.prompt, token_ids)

    model = TinyGPT(len(itos), args.block_size, args.n_layer, args.n_head,
                    args.n_embd)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {args.n_layer} layers, {args.n_head} heads, "
          f"{args.n_embd}-dim embeddings, {n_params:,} parameters "
          "(randomly initialised)")
    print(f"Sampling prompt: {prompt!r} | sampling seed: {args.seed}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                  weight_decay=0.01)

    def get_batch():
        ix = torch.randint(len(data) - args.block_size - 1, (args.batch_size,))
        x = torch.stack([data[i:i + args.block_size] for i in ix])
        y = torch.stack([data[i + 1:i + 1 + args.block_size] for i in ix])
        return x, y

    samples = []

    def sample(label, step, loss):
        print(f"\n--- sample {label} ---")
        text = generate_text(model, enc, stoi, itos, prompt, args.seed,
                             args.gen_tokens, args.temperature, stream=True)
        print("---\n")
        samples.append({"label": label, "step": step, "loss": loss,
                        "text": text})

    sample("before training (step 0)", 0, None)

    print("step | loss | elapsed time | tokens/second")
    model.train()
    rows = []
    train_time = 0.0
    tokens_seen = 0
    step = 0
    loss_val = float("nan")
    while train_time < args.seconds:
        t0 = time.perf_counter()
        x, y = get_batch()
        _, loss = model(x, y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        train_time += time.perf_counter() - t0

        step += 1
        tokens_seen += x.numel()
        loss_val = loss.item()
        tps = tokens_seen / train_time
        rows.append({"step": step, "loss": loss_val,
                     "elapsed_s": round(train_time, 3),
                     "tokens_per_s": round(tps, 1)})
        if step == 1 or step % args.print_every == 0:
            print(f"step {step:5d} | loss {loss_val:.4f} | "
                  f"elapsed {train_time:6.1f}s | tokens/s {tps:,.0f}",
                  flush=True)
        if step % args.sample_every == 0:
            sample(f"after {step} updates", step, loss_val)

    print(f"step {step:5d} | loss {loss_val:.4f} | "
          f"elapsed {train_time:6.1f}s | tokens/s {tokens_seen / train_time:,.0f}"
          "  (final)")
    sample(f"after training (final, step {step})", step, loss_val)

    # ---- Save results ----
    with open(os.path.join(args.out, "loss_log.csv"), "w", newline="",
              encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["step", "loss", "elapsed_s",
                                               "tokens_per_s"])
        writer.writeheader()
        writer.writerows(rows)

    save_plot(rows, os.path.join(args.out, "loss_plot.png"))

    with open(os.path.join(args.out, "samples.txt"), "w",
              encoding="utf-8") as f:
        f.write(f"Prompt: {prompt!r} | seed: {args.seed} | "
                f"temperature: {args.temperature}\n\n")
        for s in samples:
            loss_txt = "n/a" if s["loss"] is None else f"{s['loss']:.4f}"
            f.write(f"=== {s['label']} | loss {loss_txt} ===\n{s['text']}\n\n")
    with open(os.path.join(args.out, "samples.json"), "w",
              encoding="utf-8") as f:
        json.dump(samples, f, ensure_ascii=False, indent=2)

    torch.save({
        "model_state": model.state_dict(),
        "config": model.config,
        "itos": itos,
        "tokenizer": "o200k_base",
        "prompt": prompt,
        "sample_seed": args.seed,
        "steps": step,
        "final_loss": loss_val,
    }, os.path.join(args.out, "model.pt"))

    print(f"Saved to {args.out}/: training_log.txt, loss_log.csv, "
          "loss_plot.png, samples.txt, samples.json, model.pt")
    sys.stdout = sys.__stdout__
    log_file.close()


if __name__ == "__main__":
    main()
