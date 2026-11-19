"""Score the brain and a naive RAG baseline on golden.toml, and score the checker on hand-labeled traps.

python -I eval.py  > ../../evidence/brain-eval-2026-11-18.txt
"""
from __future__ import annotations

import sys
import tomllib
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))  # -I drops the script dir
from brain import Answer, Index, answer, build_ledger, check, extract, load_corpus, naive_answer, same, sentences

HERE = Path(__file__).resolve().parent


def stated(text: str, attr: str) -> list[str]:
    return [v for s in sentences(text) for a, v in extract(s, "2026") if a == attr]


def grade(text: str, q: dict) -> str:
    vals = stated(text, q["attr"])
    if not vals:
        return "no answer"
    if any(not any(same(v, a, q["attr"]) for a in q["accept"]) for v in vals):
        return "WRONG"
    if not all(any(same(v, r, q["attr"]) for v in vals) for r in q["require"]):
        return "INCOMPLETE"
    return "right"


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    docs = load_corpus()
    ledger = build_ledger(docs)
    index = Index(docs)
    gold = tomllib.loads((HERE / "golden.toml").read_text(encoding="utf-8"))
    print(f"Joulent brain eval. corpus: {len(docs)} docs, ledger: {len(ledger)} claims")

    print("\n== Questions: brain vs naive RAG (top BM25 sentence) ==")
    tally = {"brain": {}, "naive": {}}
    naive_wrong = naive_wrong_caught = 0
    for q in gold["question"]:
        print(f"\nQ: {q['q']}\n   gold: {q['gold']}")
        for name, ans in (("brain", answer(q["q"], ledger)), ("naive", naive_answer(q["q"], index))):
            g = grade(ans.text, q)
            fails = check(ans, ledger, docs)
            tally[name][g] = tally[name].get(g, 0) + 1
            if name == "naive" and g in ("WRONG", "INCOMPLETE"):
                naive_wrong += 1
                naive_wrong_caught += bool(fails)
            print(f"   {name}: {g:10} cites={ans.citations}")
            print(f"      text: {ans.text[:260]}")
            print(f"      checker: {[k for k, _ in fails] or 'clean'}" + "".join(f"\n        - {k}: {d}" for k, d in fails))
    for name in tally:
        print(f"\n{name}: " + ", ".join(f"{k} {v}" for k, v in sorted(tally[name].items())))
    print(f"naive answers wrong or incomplete: {naive_wrong}; checker flagged {naive_wrong_caught} of them")

    print("\n== Checker on hand-labeled answers ==")
    traps = gold["trap"]
    caught = missed = false_flags = kind_match = 0
    bad = [t for t in traps if t["kind"]]
    good = [t for t in traps if not t["kind"]]
    for t in traps:
        fails = check(Answer("", t["text"], t["cites"]), ledger, docs)
        kinds = [k for k, _ in fails]
        if t["kind"]:
            if kinds:
                caught += 1
                kind_match += t["kind"] in kinds
                verdict = "caught" + ("" if t["kind"] in kinds else f" (as {kinds}, labeled {t['kind']})")
            else:
                missed += 1
                verdict = "MISSED"
        else:
            if kinds:
                false_flags += 1
                verdict = f"FALSE FLAG {kinds}"
            else:
                verdict = "clean (correct)"
        print(f"- [{t['kind'] or 'correct'}] {t['text'][:90]} cites={t['cites']}\n    -> {verdict}"
              + "".join(f"\n       {k}: {d}" for k, d in fails))
    print(f"\nwrong answers: {len(bad)}, caught {caught}, missed {missed}, labeled kind matched {kind_match}")
    print(f"correct answers: {len(good)}, false flags {false_flags}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
