"""Download MedQA (USMLE, four options) and write it the way clinician.yaml reads it: one case per line.

    python examples/clinician/get_medqa.py                 # the test split, 1,273 cases, to data/medqa_test.jsonl
    python examples/clinician/get_medqa.py --limit 10 --out examples/clinician/data/medqa_sample.jsonl

Each line is {"id": ..., "question": "<the case>\\nA) ...\\nB) ...\\nC) ...\\nD) ...", "answer": "<letter>"}.
MedQA: Jin et al., 2021, MIT License, https://github.com/jind11/MedQA. The file read here is a
Hugging Face copy of its US test split.
"""
import argparse
import json
import os
import urllib.request

URL = "https://huggingface.co/datasets/GBaker/MedQA-USMLE-4-options/resolve/main/phrases_no_exclude_test.jsonl"
HERE = os.path.dirname(os.path.abspath(__file__))


def convert(row, index):
    """A MedQA row as a case: the question with its options, and the letter of the right one."""
    options = "\n".join(f"{letter}) {text.strip()}" for letter, text in row["options"].items())
    return {"id": f"medqa-test-{index:04d}", "question": f"{row['question'].strip()}\n{options}",
            "answer": row["answer_idx"]}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=os.path.join(HERE, "data", "medqa_test.jsonl"))
    parser.add_argument("--limit", type=int, help="only the first N cases")
    args = parser.parse_args()

    with urllib.request.urlopen(URL) as response:
        lines = response.read().decode("utf-8").splitlines()
    cases = [convert(json.loads(line), i) for i, line in enumerate(lines[:args.limit])]
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        for case in cases:
            f.write(json.dumps(case, ensure_ascii=False) + "\n")
    print(f"wrote {len(cases)} cases to {args.out}")


if __name__ == "__main__":
    main()
