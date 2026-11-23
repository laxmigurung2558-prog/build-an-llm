"""Pick a short sentence that appears exactly in fineweb_corpus.txt, use its
first few words as the prompt, and generate 100 completions.

Run:  python sample_corpus.py
Writes corpus_completions.txt.
"""

import argparse
import re

import tiktoken

from model import generate_text, load_checkpoint, missing_tokens

# Common sentence openings, in order of preference.
PREFERRED_OPENINGS = ["The", "It", "This", "In", "A", "We", "If", "I", "They"]


def candidate_sentences(corpus, min_words, max_words):
    """Split the corpus into sentences and keep short, clean ones."""
    sentences = re.split(r"(?<=[.!?])\s+", corpus)
    keep = []
    for s in sentences:
        s = s.strip()
        words = s.split()
        if not (min_words <= len(words) <= max_words):
            continue
        if not re.match(r"^[A-Z][a-z]*$", words[0]):  # plain capitalised first word
            continue
        if s in corpus:  # must appear exactly in the corpus
            keep.append(s)
    return keep


def preference(sentence):
    first = sentence.split()[0]
    return (PREFERRED_OPENINGS.index(first)
            if first in PREFERRED_OPENINGS else len(PREFERRED_OPENINGS))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--corpus", default="fineweb_corpus.txt")
    p.add_argument("--checkpoint", default="results/model.pt")
    p.add_argument("--begin-words", type=int, default=3)
    p.add_argument("--min-words", type=int, default=5)
    p.add_argument("--max-words", type=int, default=15)
    p.add_argument("--n", type=int, default=100)
    p.add_argument("--tokens", type=int, default=30)
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--out", default="corpus_completions.txt")
    args = p.parse_args()

    enc = tiktoken.get_encoding("o200k_base")
    model, stoi, itos, _ = load_checkpoint(args.checkpoint)
    with open(args.corpus, encoding="utf-8") as f:
        corpus = f.read()

    candidates = sorted(candidate_sentences(corpus, args.min_words,
                                            args.max_words), key=preference)
    print(f"Found {len(candidates)} short candidate sentences in the corpus.")

    chosen = beginning = None
    for sentence in candidates:
        words = " ".join(sentence.split()[:args.begin_words])
        # Inside the corpus most sentences follow a space, so the same words
        # may be tokenized with a leading space; try both forms.
        for form in (words, " " + words):
            _, missing = missing_tokens(enc, stoi, form)
            if missing:
                print(f"  skip {form!r}: tokens not in vocabulary {missing}")
                continue
            chosen, beginning = sentence, form
            break
        if chosen:
            break

    if not chosen:
        raise SystemExit("No suitable sentence found. Try --begin-words 2.")

    ids, _ = missing_tokens(enc, stoi, beginning)
    print(f"Chosen sentence: {chosen!r}")
    print(f"Beginning used:  {beginning!r} -> tokens {ids} "
          "(all in the model's vocabulary)")

    exact = 0
    lines = []
    for k in range(args.n):
        text = generate_text(model, enc, stoi, itos, beginning, seed=k,
                             max_new_tokens=args.tokens,
                             temperature=args.temperature, stream=False)
        if text.strip().startswith(chosen):
            exact += 1
        line = f"{k + 1:3d}. {text.strip()}".replace("\n", " ")
        print(line)
        lines.append(line)

    with open(args.out, "w", encoding="utf-8") as f:
        f.write(f"Chosen sentence (appears exactly in the corpus): {chosen}\n")
        f.write(f"Sentence beginning used as prompt: {beginning!r}\n")
        f.write(f"Settings: {args.n} completions, {args.tokens} new tokens, "
                f"temperature {args.temperature}, seeds 0-{args.n - 1}\n")
        f.write(f"Completions that reproduce the full original sentence: "
                f"{exact}/{args.n}\n\n")
        f.write("\n".join(lines) + "\n")
    print(f"\n{exact}/{args.n} completions reproduce the full original sentence.")
    print(f"Saved to {args.out}")


if __name__ == "__main__":
    main()
