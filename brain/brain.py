"""A small company brain over Joulent's public corpus, plus a checker for confident wrong answers.

Stdlib only, no model calls. Layers:
  1. corpus:   documents with publisher, date and tier (company | regulator | news), corpus.toml
  2. ledger:   every claim the corpus makes about a tracked fact, with its exact sentence (extractors)
  3. answer:   route a question to a fact, resolve the ledger into a dated, cited answer that says
               when sources disagree or when a newer source changed the picture
  4. check:    grade ANY answer (ours, a naive RAG baseline, an LLM's) for four failure classes:
               UNGROUNDED   a claim the cited sources do not make
               CONTRADICTED a value the weight of sources disagrees with
               STALE        a value a newer source has changed, with no mention of the change
               ENTITY_MIXUP a value that belongs to a different entity (campus load vs plant capacity)

Usage:
  python -I brain.py ask "When is the Kilby final investment decision?"
  python -I brain.py ledger
"""
from __future__ import annotations

import math
import re
import sys
import tomllib
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path

HERE = Path(__file__).resolve().parent
TEXT = HERE.parent / "corpus" / "text"
TIER_WEIGHT = {"regulator": 3, "company": 3, "news": 1}
TEMPORAL = {"ppa_holder", "joulent_kilby_stake", "fid_year"}  # facts that change over time
TOLERANCE = 0.02  # numeric values within 2% are the same fact (2.67 GW reported as 2.7 GW)


@dataclass
class Doc:
    id: str
    publisher: str
    date: str
    tier: str
    url: str
    text: str


@dataclass(frozen=True)
class Claim:
    attr: str
    value: str
    doc: str
    date: str
    tier: str
    quote: str


@dataclass
class Answer:
    question: str
    text: str
    citations: list[str] = field(default_factory=list)


# ---------------------------------------------------------------- corpus

def load_corpus(manifest: Path = HERE / "corpus.toml", text_dir: Path = TEXT) -> dict[str, Doc]:
    docs = {}
    for d in tomllib.loads(manifest.read_text(encoding="utf-8"))["doc"]:
        text = (text_dir / d["file"]).read_text(encoding="utf-8", errors="replace")
        docs[d["id"]] = Doc(d["id"], d["publisher"], d["date"], d["tier"], d["url"], text)
    return docs


SENT_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z\"(*])|\n+")


def sentences(text: str) -> list[str]:
    return [s.strip(" \t*-") for s in SENT_SPLIT.split(text)
            if len(s.strip()) > 3 and not s.lstrip().startswith("SOURCE:")]


# ---------------------------------------------------------------- extractors (the ledger)

GW = re.compile(r"(nearly|about|approximately|~)?\s*(\d+(?:\.\d+)?)\s*-?\s*(?:GW|gigawatts?)\b", re.I)
KILBY_CONTEXT = re.compile(r"kilby|pecos|reeves|microsoft|turbine", re.I)
CAMPUS_LOAD = re.compile(r"commits to a", re.I)
YEAR = re.compile(r"\b(20[2-3]\d)\b|\b(?:end of|later) this year\b", re.I)
FID = re.compile(r"final investment decision|\bFID\b", re.I)
POSSIBLE = re.compile(r"(?:slip\w*\b[^.]*?\bto|move into)\s*$", re.I)
FIRST_POWER = re.compile(r"first power|receiving power|begin flowing|produce power|power delivery", re.I)


def _years_after(s: str, start: int, doc_year: str, window: int = 400):
    for m in YEAR.finditer(s, start, min(len(s), start + window)):
        yield m, (m.group(1) or doc_year)


def extract(sentence: str, doc_year: str) -> list[tuple[str, str]]:
    """(attr, value) pairs one sentence asserts. Same code reads the corpus and the answers."""
    out: list[tuple[str, str]] = []
    s = sentence
    if KILBY_CONTEXT.search(s) or re.search(r"\bplant\b|power facility|generation campus", s, re.I):
        for m in GW.finditer(s):
            if re.search(r"multi-?$", s[:m.start()], re.I):
                continue
            attr = "campus_load_gw" if CAMPUS_LOAD.search(s) else "plant_capacity_gw"
            out.append((attr, m.group(2)))
    f = FID.search(s)
    if f:
        for m, y in _years_after(s, f.end(), doc_year):
            # a year tied to "first power" belongs to that fact, not the decision
            if FIRST_POWER.search(s[f.end():m.start()]):
                break
            out.append(("fid_year", y + "?" if POSSIBLE.search(s[:m.start()]) else y))
    p = FIRST_POWER.search(s)
    if p:
        for m, y in _years_after(s, p.end(), doc_year, 60):
            out.append(("first_power_year", y))
            break
    if re.search(r"power purchase agreement|power deal|\bPPA\b", s, re.I) and re.search(r"Energy Forge One", s):
        if re.search(r"wholly owned Chevron subsidiary|\(Chevron subsidiary\)", s, re.I):
            out.append(("ppa_holder", "Chevron subsidiary"))
        if re.search(r"50/50 joint venture between Chevron and Joulent|Chevron-Joulent 50/50 JV", s, re.I):
            out.append(("ppa_holder", "Chevron-Joulent 50/50 JV"))
    for m in re.finditer(r"natural gas produced from (?:the energy company )?(\w+)'s fields"
                         r"|gas from (\w+)'s existing Permian", s):
        out.append(("gas_source_owner", m.group(1) or m.group(2)))
    if re.search(r"Joulent", s) and re.search(r"50% ownership (?:interest|stake)", s):
        out.append(("joulent_kilby_stake", "50%"))
    if re.search(r"Kilby", s) and re.search(r"50/50 (?:partnership|joint venture) with Chevron", s):
        out.append(("joulent_kilby_stake", "50%"))
    if re.search(r"option to take on half", s):
        out.append(("joulent_kilby_stake", "option on 50%"))
    m = re.search(r"\$1\.75 billion to secure a (\d+)% interest", s)
    if m:
        out.append(("ngv_stake", m.group(1) + "%"))
    if re.search(r"not disclosed (?:a cost estimate|the project's cost)|cost[^.]*not disclosed", s, re.I):
        out.append(("project_cost", "undisclosed"))
    m = re.search(r"\$(\d+) billion (?:co-development|for the project|total cost)|"
                  r"(?:total )?costs? (?:at|of) (?:roughly |about )?\$(\d+) billion", s, re.I)
    if m:
        est = " estimate" if re.search(r"estimat", s, re.I) else ""
        out.append(("project_cost", "$" + (m.group(1) or m.group(2)) + "B" + est))
    return out


def build_ledger(docs: dict[str, Doc]) -> list[Claim]:
    ledger = []
    for d in docs.values():
        for s in sentences(d.text):
            for attr, value in extract(s, d.date[:4]):
                ledger.append(Claim(attr, value, d.id, d.date, d.tier, s[:300]))
    return list(dict.fromkeys(ledger))


def same(a: str, b: str, attr: str = "") -> bool:
    """Equal values; numeric tolerance only for GW figures (2.67 reported as 2.7), never for years."""
    if not attr.endswith("_gw"):
        return a == b
    try:
        x, y = float(a), float(b)
    except ValueError:
        return a == b
    return abs(x - y) <= TOLERANCE * max(x, y)


# ---------------------------------------------------------------- retrieval (BM25 over sentences)

TOKEN = re.compile(r"[a-z0-9.]+")


def tokens(s: str) -> list[str]:
    return [t.strip(".") for t in TOKEN.findall(s.lower()) if t.strip(".")]


class Index:
    def __init__(self, docs: dict[str, Doc]):
        self.rows = [(d.id, s) for d in docs.values() for s in sentences(d.text)]
        self.toks = [tokens(s) for _, s in self.rows]
        self.df = Counter(t for ts in self.toks for t in set(ts))
        self.avg = sum(map(len, self.toks)) / max(1, len(self.toks))

    def search(self, q: str, k: int = 5) -> list[tuple[float, str, str]]:
        qt, n, out = tokens(q), len(self.rows), []
        for (doc, s), ts in zip(self.rows, self.toks):
            tf, score = Counter(ts), 0.0
            for t in qt:
                if t in tf:
                    idf = math.log(1 + (n - self.df[t] + 0.5) / (self.df[t] + 0.5))
                    score += idf * tf[t] * 2.2 / (tf[t] + 1.2 * (0.25 + 0.75 * len(ts) / self.avg))
            if score:
                out.append((score, doc, s))
        return sorted(out, reverse=True)[:k]


# ---------------------------------------------------------------- answering

ROUTES = [
    ("campus_load_gw", r"campus (?:load|size)|how big is microsoft|data center (?:load|size)"),
    ("plant_capacity_gw", r"capacity|how many (?:gw|gigawatts)|how big is (?:the )?(?:kilby|plant)"),
    ("fid_year", r"final investment decision|\bfid\b"),
    ("first_power_year", r"first power|start (?:delivering|receiving) power|online"),
    ("ppa_holder", r"\bppa\b|power purchase|who (?:holds|signed)"),
    ("gas_source_owner", r"whose gas|gas (?:come|source)|fuel"),
    ("joulent_kilby_stake", r"joulent'?s? (?:stake|share|ownership)|how much of kilby"),
    ("ngv_stake", r"national grid.*(?:stake|interest|percent)"),
    ("project_cost", r"cost|how much will kilby"),
]
LABEL = {
    "plant_capacity_gw": "Kilby generation capacity (GW)",
    "campus_load_gw": "Microsoft Pecos campus size (GW)",
    "fid_year": "Kilby final investment decision (year)",
    "first_power_year": "Kilby first power (year)",
    "ppa_holder": "Holder of the Microsoft PPA",
    "gas_source_owner": "Owner of the gas that fuels Kilby",
    "joulent_kilby_stake": "Joulent's stake in Kilby",
    "ngv_stake": "National Grid Ventures' stake in Joulent",
    "project_cost": "Kilby project cost",
}
PHRASE = {  # how an answer states each value so the same extractors can read it back
    "plant_capacity_gw": "Kilby is a {v} GW plant",
    "campus_load_gw": "Microsoft commits to a {v} GW Pecos campus",
    "fid_year": "the Kilby final investment decision is expected in {v}",
    "first_power_year": "Kilby first power is targeted for {v}",
    "ppa_holder": "Energy Forge One holds the 20-year power purchase agreement ({v})",
    "gas_source_owner": "Kilby burns natural gas from {v}'s existing Permian production",
    "joulent_kilby_stake": "Joulent holds a 50% ownership stake in Kilby ({v})",
    "ngv_stake": "National Grid Ventures paid $1.75 billion to secure a {n} interest",
    "project_cost": "Kilby's cost: {v}",
}
PHRASE_SPECIAL = {
    ("project_cost", "undisclosed"): "Chevron has not disclosed the project's cost",
}


def route(question: str) -> str | None:
    for attr, pat in ROUTES:
        if re.search(pat, question, re.I):
            return attr
    return None


def resolve(ledger: list[Claim], attr: str) -> dict:
    """Weigh the ledger for one fact: winning value, dissent, and changes over time."""
    claims = [c for c in ledger if c.attr == attr]
    groups: list[list[Claim]] = []
    for c in claims:
        for g in groups:
            if same(g[0].value, c.value, attr):
                g.append(c)
                break
        else:
            groups.append([c])

    def weight(g):  # one vote per document, weighted by tier
        return sum(TIER_WEIGHT[c.tier] for c in {c.doc: c for c in g}.values())

    if attr in TEMPORAL:  # facts that change: the newest source wins, weight breaks ties
        groups.sort(key=lambda g: (max(c.date for c in g), weight(g)), reverse=True)
    else:
        groups.sort(key=lambda g: (weight(g), max(c.date for c in g)), reverse=True)
    return {"attr": attr, "groups": groups, "weights": [weight(g) for g in groups]}


def _fmt(v: str) -> str:
    return v.rstrip("?")


def answer(question: str, ledger: list[Claim]) -> Answer:
    attr = route(question)
    if attr is None:
        return Answer(question, "I don't track that fact yet; no answer rather than a guess.", [])
    r = resolve(ledger, attr)
    if not r["groups"]:
        return Answer(question, f"No source in the corpus states {LABEL[attr]}.", [])
    top = r["groups"][0]
    v = top[0].value
    cites = sorted({c.doc for c in top}, key=lambda d: d)
    phrase = PHRASE_SPECIAL.get((attr, v), PHRASE[attr])
    lines = [phrase.format(v=_fmt(v), n=v) + f" [{', '.join(cites)}]."]
    newest_top = max(c.date for c in top)
    for g, w in zip(r["groups"][1:], r["weights"][1:]):
        gv, gd = g[0].value, max(c.date for c in g)
        docs = sorted({c.doc for c in g})
        if gv.endswith("?"):
            lines.append(f"Since {gd}, reporting says the final investment decision could slip to {_fmt(gv)} "
                         f"[{', '.join(docs)}]; no decision is announced in the corpus.")
        elif attr in TEMPORAL and gd < newest_top:
            lines.append(f"Earlier ({gd}) {', '.join(docs)} said {_fmt(gv)}; superseded by the newer source.")
        elif gd > newest_top:
            lines.append(f"Newer source ({gd}) says {_fmt(gv)} [{', '.join(docs)}]; check before relying on it.")
        else:
            lines.append(f"Sources disagree: {', '.join(docs)} say {_fmt(gv)} (weight {w} vs {r['weights'][0]}).")
        cites += [d for d in docs if d not in cites]
    return Answer(question, " ".join(lines), cites)


# ---------------------------------------------------------------- checking any answer

def check(ans: Answer, ledger: list[Claim], docs: dict[str, Doc]) -> list[tuple[str, str]]:
    """Failures in one answer: [(kind, detail)]. Empty list = clean."""
    if not ans.citations:
        claims_in = [a for s in sentences(ans.text) for a in extract(s, "2026")]
        return [("UNGROUNDED", "no citations")] if claims_in else []
    fails = []
    cited = set(ans.citations)
    unknown = cited - set(docs)
    if unknown:
        fails.append(("UNGROUNDED", f"cites unknown docs {sorted(unknown)}"))
    stated = [a for s in sentences(ans.text) for a in extract(s, "2026")]
    stated_vals = defaultdict(list)
    for attr, v in stated:
        stated_vals[attr].append(v)
    for attr, v in stated:
        support = [c for c in ledger if c.attr == attr and same(c.value, v, attr)]
        if not any(c.doc in cited for c in support):
            other = [c for c in ledger if c.attr == "campus_load_gw" and same(c.value, v, attr)] \
                if attr == "plant_capacity_gw" else []
            if other:
                fails.append(("ENTITY_MIXUP", f"{v} GW is the campus size ({other[0].doc}), not plant capacity"))
                continue
            fails.append(("UNGROUNDED", f"{attr}={v} not stated by any cited source"))
        r = resolve(ledger, attr)
        if r["groups"] and not same(r["groups"][0][0].value, v, attr) and not v.endswith("?"):
            win = r["groups"][0][0].value
            if not any(same(win, x, attr) for x in stated_vals[attr]):
                fails.append(("CONTRADICTED", f"{attr}={v}; weight of sources says {win} "
                                              f"({r['weights'][0]} vs {dict(zip([g[0].value for g in r['groups']], r['weights'])).get(v, 0)})"))
        if support and not v.endswith("?") and attr in TEMPORAL:  # only facts that change can go stale
            own = [c for c in support if c.doc in cited] or support  # judge the sources the answer relied on
            newest_support = max(c.date for c in own)
            newer = [c for c in ledger if c.attr == attr and not same(c.value, v, attr) and c.date > newest_support]
            newer = [c for c in newer if not any(same(c.value, x, attr) for x in stated_vals[attr])]
            if newer:
                c = max(newer, key=lambda c: c.date)
                fails.append(("STALE", f"{attr}={v} (newest support {newest_support}); {c.doc} ({c.date}) says {c.value}"))
    return list(dict.fromkeys(fails))


def naive_answer(question: str, index: Index) -> Answer:
    """Baseline: top BM25 sentence, quoted as the answer, one citation. What a quick RAG demo does."""
    hits = index.search(question, 1)
    if not hits:
        return Answer(question, "", [])
    _, doc, s = hits[0]
    return Answer(question, s, [doc])


def main(argv: list[str]) -> int:
    docs = load_corpus()
    ledger = build_ledger(docs)
    if argv[:1] == ["ledger"]:
        for c in sorted(ledger, key=lambda c: (c.attr, c.date)):
            print(f"{c.attr:22} {c.value:28} {c.date} {c.tier:9} {c.doc}")
        return 0
    if argv[:1] == ["ask"] and len(argv) > 1:
        a = answer(" ".join(argv[1:]), ledger)
        print(a.text)
        print("check:", check(a, ledger, docs) or "clean")
        return 0
    print(__doc__)
    return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
