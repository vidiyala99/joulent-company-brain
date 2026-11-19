# joulent-company-brain

A small "company brain" over Joulent's public record, plus an eval that catches confident wrong answers.

Built in an evening as a working sketch of the AI Engineer, Company Brain problem: one place to find what a company
knows and has decided, where every answer is cited and wrong answers are caught before anyone relies on them.
Everything here comes from public sources (joulent.com, the National Grid Ventures factsheet it links, TCEQ permit
notices, news coverage). Standard library Python only, no model calls, no API keys.

## What it does

1. **Corpus** (`corpus/text/`, `brain/corpus.toml`): 19 public documents, each with publisher, date and a source
   type (company, regulator, news). News files are short verbatim excerpts with their URL and fetch date.
2. **Claim ledger**: rule-based extractors pull every claim the corpus makes about 9 tracked facts on Project Kilby
   (plant capacity, Microsoft campus size, final investment decision date, first power, PPA holder, gas source,
   Joulent's stake, National Grid's stake, cost), each with the exact sentence it came from.
3. **Answerer**: routes a question to a fact, weighs sources (company and regulator above news), lets the newest
   source win for facts that change over time (who holds the PPA, Joulent's stake, the FID date), and says when
   sources disagree instead of picking one silently. It refuses questions it does not track.
4. **Checker**: grades any answer (this brain's, a retrieval baseline's, an LLM's) for four failures:
   - `UNGROUNDED`: a claim the cited sources do not make
   - `CONTRADICTED`: a value the weight of sources disagrees with
   - `STALE`: a value a newer source has changed, with no mention of the change
   - `ENTITY_MIXUP`: a value that belongs to a different thing (the 2 GW Microsoft campus vs the 2.67 GW plant)

## Run

```
cd brain
python -I brain.py ask "When is the Kilby final investment decision?"
python -I brain.py ledger
python -I eval.py
python -I -m unittest test_brain
```

Example:

```
$ python -I brain.py ask "When is the Kilby final investment decision?"
the Kilby final investment decision is expected in 2026 [news-aea, news-cutrer, news-qz, news-startupfortune, news-tnw, ngv-factsheet].
Since 2026-10-01, reporting says the final investment decision could slip to 2027 [news-cutrer, news-tnw]; no decision is announced in the corpus.
check: clean
```

## Results (`results/eval-output.txt`)

| | |
|---|---|
| Brain, 9 golden questions | 9 right |
| Naive retrieval (top BM25 sentence quoted as the answer) | 4 right, 4 no answer, 1 incomplete |
| Checker, 10 hand-labeled wrong answers | 10 caught |
| Checker, 10 hand-labeled correct answers | 0 false alarms |

The naive baseline's one bad answer is the dangerous kind: asked about the FID, it quotes the July factsheet
("expected in 2026") with no sign of the October reporting that it could slip to 2027. The checker marks it `STALE`.

**The first eval round found bugs in the checker itself:** it caught 9 of 10 and raised 2 false alarms.
The miss: a 2% tolerance meant for "2.67 GW reported as 2.7 GW" also let "2027" match "2028". The false alarms:
`STALE` fired on fixed facts whenever a newer, weaker source disagreed. Both fixes carry regression tests.
The hand labels in `brain/golden.toml` were committed before the checker was scored.

**Caveat:** the same person wrote the extractors and the hand-labeled answers, so this is an in-distribution test.
The honest claim is "it catches these 10 failure cases and stays quiet on these 10 correct answers", not a general
accuracy rate. The extractors are regexes over 9 facts; a production brain would extract claims with a model and
calibrate that extractor against human labels the same way.

## What the corpus itself shows

- The factsheet linked from joulent.com's July 1 release says the Kilby final investment decision "is expected in
  2026". Since October 1, reporting (Semafor, via The Next Web) says it could slip to 2027 after Texas paused data
  center permits on September 21.
- Sources disagree on basic facts: People says the gas comes from "Joulent's fields" (Quartz and AEA: Chevron's
  Permian production); the June launch release calls Energy Forge One "a wholly owned Chevron subsidiary", October
  reporting calls it "a 50/50 joint venture between Chevron and Joulent"; capacity appears as 2.67, 2.7 and "nearly
  3" GW; one article's summary states a "$7 billion" cost that its own body says is undisclosed.

## Sources

Every document's URL, publisher and date is in `brain/corpus.toml`; news files start with a `SOURCE:` line. Excerpts
are quoted for analysis. Fetched 2026-11-18.

Author: Aakash Vidiyala.
