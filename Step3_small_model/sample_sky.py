"""Load the trained model and complete "The sky is" 100 times.

Run:  python sample_sky.py
Writes sky_completions.txt. If the prompt contains tokens the tiny model
never saw during training, it explains which ones and stops.
"""

import argparse
import sys

import tiktoken

from model import generate_text, load_checkpoint, missing_tokens


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", default="results/model.pt")
    p.add_argument("--prompt", default="The sky is")
    p.add_argument("--n", type=int, default=100)
    p.add_argument("--tokens", type=int, default=30)
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--out", default="sky_completions.txt")
    args = p.parse_args()

    enc = tiktoken.get_encoding("o200k_base")
    model, stoi, itos, _ = load_checkpoint(args.checkpoint)

    ids, missing = missing_tokens(enc, stoi, args.prompt)
    pieces = [repr(enc.decode([i])) for i in ids]
    print(f"Prompt {args.prompt!r} -> tokens {ids} = {' '.join(pieces)}")

    if missing:
        missing_txt = ", ".join(f"{i} ({enc.decode([i])!r})" for i in missing)
        msg = (f"FAILED: the prompt {args.prompt!r} contains token(s) that never "
               f"appeared in the training corpus, so the model has no "
               f"embedding for them: {missing_txt}\n"
               f"The model's vocabulary only has {len(itos)} tokens (those in "
               f"fineweb_corpus.txt).\n")
        print(msg)
        with open(args.out, "w", encoding="utf-8") as f:
            f.write(msg)
        print(f"Saved this result to {args.out}")
        sys.exit(1)

    with open(args.out, "w", encoding="utf-8") as f:
        f.write(f"Prompt: {args.prompt!r} | {args.n} completions | "
                f"{args.tokens} new tokens | temperature {args.temperature}\n\n")
        for k in range(args.n):
            text = generate_text(model, enc, stoi, itos, args.prompt, seed=k,
                                 max_new_tokens=args.tokens,
                                 temperature=args.temperature, stream=False)
            line = f"{k + 1:3d}. {text}"
            print(line)
            f.write(line.replace("\n", " ") + "\n")
    print(f"Saved {args.n} completions to {args.out}")


if __name__ == "__main__":
    main()
