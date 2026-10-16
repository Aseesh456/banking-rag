"""Measure retrieval quality of the RAG pipeline on a small question set.

Usage:
    python evaluate.py --docs eval_docs --questions eval_set.json --k 4
    python evaluate.py --docs eval_docs --questions eval_set.json --chunk-size 300   # try another chunk size
    python evaluate.py --docs eval_docs --questions eval_set.json --answers   # also grade Gemini's answers

eval_set.json is a list of {"question": "...", "keywords": ["word1", "word2"]}.
A retrieved chunk "contains the answer" when ALL keywords appear in it (case-insensitive).
Metrics: Hit@k (answer found in top-k chunks) and MRR (how high the first correct chunk ranks).
The evaluation uses its own index folder, so your normal knowledge base is untouched.
"""
import argparse
import json
import os
import shutil

import vector_store
from ingest import ingest_file

EVAL_INDEX = "eval_index"


def build_index(docs_dir):
    shutil.rmtree(EVAL_INDEX, ignore_errors=True)
    total = 0
    for name in sorted(os.listdir(docs_dir)):
        if name.lower().endswith((".pdf", ".png", ".jpg", ".jpeg")):
            n = ingest_file(os.path.join(docs_dir, name))
            print(f"indexed {name}: {n} chunks")
            total += n
    return total


def contains_all(text, keywords):
    t = text.lower()
    return all(k.lower() in t for k in keywords)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--docs", help="folder with the PDFs/images to index (rebuilds the eval index)")
    p.add_argument("--questions", required=True)
    p.add_argument("--k", type=int, default=4)
    p.add_argument("--chunk-size", type=int, help="override chunk size when rebuilding the index (overlap = size/6)")
    p.add_argument("--answers", action="store_true", help="also ask Gemini and check keywords in its answer")
    args = p.parse_args()

    vector_store.INDEX_DIR = EVAL_INDEX  # keep evaluation separate from the real index
    if args.chunk_size:
        _orig = vector_store.chunk_text
        vector_store.chunk_text = lambda t: _orig(t, chunk_size=args.chunk_size, overlap=args.chunk_size // 6)
    if args.docs:
        build_index(args.docs)
    if not vector_store.index_exists():
        raise SystemExit("No eval index found. Run with --docs <folder> first.")

    with open(args.questions) as f:
        items = json.load(f)

    if args.answers:
        from rag_engine import answer_question

    hits, rr_sum, ans_ok, rows = 0, 0.0, 0, []
    for item in items:
        q, kws = item["question"], item["keywords"]
        results = vector_store.search(q, k=args.k)
        rank = next((i + 1 for i, r in enumerate(results) if contains_all(r[0], kws)), None)
        hit = rank is not None
        hits += hit
        rr_sum += 1 / rank if rank else 0
        row = {"question": q, "rank_of_first_correct_chunk": rank}
        if args.answers:
            ans = answer_question(q, k=args.k)["answer"]
            row["answer_ok"] = contains_all(ans, kws)
            ans_ok += row["answer_ok"]
        rows.append(row)
        print(f'{"HIT " if hit else "MISS"} rank={rank}  {q}')

    n = len(items)
    print("\n--- summary ---")
    print(f"questions: {n}   k: {args.k}")
    print(f"Hit@{args.k}: {hits}/{n} = {hits / n:.0%}")
    print(f"MRR: {rr_sum / n:.2f}")
    if args.answers:
        print(f"Answers containing expected facts: {ans_ok}/{n} = {ans_ok / n:.0%}")

    with open("eval_results.json", "w") as f:
        json.dump({"k": args.k, "hit_rate": hits / n, "mrr": rr_sum / n, "rows": rows}, f, indent=2)


if __name__ == "__main__":
    main()
