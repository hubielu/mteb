"""Run MoCa on the four ViDoRe-v2 tasks in mteb and print a table for the PR.

The paper (arXiv:2506.23115, Table 2) reports NDCG@5 on ViDoRe-v2, which is also the
`main_score` of these tasks in mteb, so the numbers are directly comparable. mteb splits
each dataset into english/french/spanish/german subsets over a shared corpus, whereas the
paper reports an English column and a pooled multilingual column, so compare the english
subset to the English column and treat the rest as directional.

Usage:
    python run_moca_vidore.py --batch-size 4
    python run_moca_vidore.py --tasks Vidore2ESGReportsHLRetrieval   # single cheap task
"""

from __future__ import annotations

import argparse
import json

import mteb

TASKS = [
    "Vidore2ESGReportsHLRetrieval",
    "Vidore2ESGReportsRetrieval",
    "Vidore2EconomicsReportsRetrieval",
    "Vidore2BioMedicalLecturesRetrieval",
]

# MoCa-3B English columns from Table 2 of the paper, keyed by mteb task name.
PAPER_3B = {
    "Vidore2ESGReportsHLRetrieval": ("ESG_Human", 63.3),
    "Vidore2ESGReportsRetrieval": ("ESG_Syn", 58.3),
    "Vidore2EconomicsReportsRetrieval": ("Eco", 62.8),
    "Vidore2BioMedicalLecturesRetrieval": ("Bio", 62.5),
}


def summarize(results) -> None:
    rows = []
    for res in results:
        try:
            for split, entries in res.scores.items():
                for entry in entries:
                    rows.append(
                        {
                            "task": res.task_name,
                            "split": split,
                            "subset": entry.get("hf_subset", "default"),
                            "ndcg_at_5": entry.get(
                                "ndcg_at_5", entry.get("main_score")
                            ),
                        }
                    )
        except Exception as err:  # noqa: BLE001 - never lose a paid run to a print bug
            print(f"could not parse scores for {res}: {err}")

    print(f"\n{'task':<38} {'subset':<10} {'ndcg@5':>7}")
    for row in rows:
        print(f"{row['task']:<38} {row['subset']:<10} {100 * row['ndcg_at_5']:>7.1f}")

    print(f"\n{'task':<38} {'paper':<12} {'paper':>7} {'ours (eng)':>11} {'delta':>7}")
    for task, (column, expected) in PAPER_3B.items():
        ours = next(
            (
                100 * r["ndcg_at_5"]
                for r in rows
                if r["task"] == task and r["subset"] == "english"
            ),
            None,
        )
        if ours is None:
            continue
        print(
            f"{task:<38} {column:<12} {expected:>7.1f} {ours:>11.1f} {ours - expected:>+7.1f}"
        )

    with open("moca_vidore_summary.json", "w") as fh:
        json.dump(rows, fh, indent=2)
    print("\nwrote moca_vidore_summary.json")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="moca-embed/MoCa-Qwen25VL-3B")
    parser.add_argument("--tasks", nargs="*", default=TASKS)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--output-folder", default="results")
    args = parser.parse_args()

    model = mteb.get_model(args.model)
    tasks = mteb.get_tasks(tasks=args.tasks)
    results = mteb.evaluate(
        model,
        tasks=tasks,
        output_folder=args.output_folder,
        encode_kwargs={"batch_size": args.batch_size},
    )
    summarize(results)


if __name__ == "__main__":
    main()
