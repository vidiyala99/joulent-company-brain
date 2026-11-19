# joulent-company-brain

A small company brain over Joulent's public record, with an eval that catches confident wrong answers.
Built from public sources only (joulent.com, the National Grid Ventures factsheet it links, TCEQ notices, news),
fetched 2026-11-18. Standard library Python, no model calls, no API keys.

## What I found

1. **A linked decision went stale.** The factsheet joulent.com's July 1 release links says the Kilby final
   investment decision "is expected in 2026". Since October 1, reporting (Semafor, via The Next Web) says it could
   slip to 2027 after Texas paused data center permits on September 21. No decision is announced in the corpus.
2. **Sources disagree on basic facts.** Gas source: People says "Joulent's fields"; Quartz and AEA say Chevron's
   Permian production. PPA holder: the June launch release calls Energy Forge One "a wholly owned Chevron
   subsidiary"; October reporting calls it "a 50/50 joint venture between Chevron and Joulent". Capacity appears as
   2.67, 2.7 and "nearly 3" GW.
3. **An unpinned script runs on every page.** joulent.com loads `unpkg.com/split-type` with no version and no
   integrity hash, including on the Careers form (name, email, phone, LinkedIn). unpkg serves the newest npm release,
   so a new publish or a hijacked maintainer account would run on the site within minutes. Fix: pin
   `split-type@0.3.4` with an integrity hash, or self-host it.

## Results (`results/eval-output.txt`)

| | |
|---|---|
| Brain, 9 golden questions | 9 right |
| Naive retrieval (top BM25 sentence as the answer) | 4 right, 4 no answer, 1 incomplete |
| Checker, 10 hand-labeled wrong answers | 10 caught |
| Checker, 10 hand-labeled correct answers | 0 false alarms |

The baseline's incomplete answer is the dangerous kind: asked about the FID, it quotes "expected in 2026" with no sign of
the October reporting. The checker marks it `STALE`.

Round 1 found bugs in the checker itself (9 of 10 caught, 2 false alarms): a tolerance meant for "2.67 vs 2.7 GW"
let "2027" match "2028", and `STALE` fired on fixed facts. Both fixes carry regression tests. The hand labels in
`brain/golden.toml` were committed before scoring.

**Caveat:** I wrote both the extractors and the labels, so this is in-distribution: it shows the method, not a
general accuracy rate. A production brain would extract claims with a model and calibrate that against human labels
the same way.

## How it works

- **Corpus** (`corpus/text/`, `brain/corpus.toml`): 19 documents, each with publisher, date and source type.
- **Claim ledger**: every claim about 9 tracked Kilby facts, with the exact sentence it came from.
- **Answerer**: weighs company and regulator sources above news, lets the newest source win for facts that change,
  says when sources disagree, and refuses questions it does not track.
- **Checker**: grades any answer for `UNGROUNDED`, `CONTRADICTED`, `STALE` and `ENTITY_MIXUP` (the 2 GW Microsoft
  campus vs the 2.67 GW plant).

```
cd brain
python -I brain.py ask "When is the Kilby final investment decision?"
python -I eval.py
python -I -m unittest test_brain
```

Author: Aakash Vidiyala.
