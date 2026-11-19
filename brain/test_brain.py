"""Planted-flaw tests for the brain checker. python -I -m unittest discover -s . (from build/brain)."""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from brain import Answer, answer, build_ledger, check, extract, load_corpus, same  # noqa: E402

DOCS = load_corpus()
LEDGER = build_ledger(DOCS)


def kinds(text, cites):
    return [k for k, _ in check(Answer("", text, cites), LEDGER, DOCS)]


class Tolerance(unittest.TestCase):
    def test_gw_rounding_is_the_same_fact(self):
        self.assertTrue(same("2.67", "2.7", "plant_capacity_gw"))

    def test_years_are_never_rounded(self):  # bug found in eval round 1: 2027 matched 2028
        self.assertFalse(same("2027", "2028", "first_power_year"))


class Extractors(unittest.TestCase):
    def test_fid_and_first_power_in_one_sentence(self):
        got = extract("A Final Investment Decision on Project Kilby is expected in 2026, with first power delivery in 2028.", "2026")
        self.assertIn(("fid_year", "2026"), got)
        self.assertIn(("first_power_year", "2028"), got)
        self.assertNotIn(("fid_year", "2028"), got)

    def test_possible_slip_is_marked(self):
        got = extract("A final investment decision could slip from the end of this year to 2027.", "2026")
        self.assertIn(("fid_year", "2027?"), got)
        self.assertIn(("fid_year", "2026"), got)

    def test_bio_gigawatts_are_not_kilby(self):
        self.assertEqual(extract("He built over 5 GW of new traditional power generation projects.", "2026"), [])


class PlantedFlaws(unittest.TestCase):
    def test_no_citation(self):
        self.assertIn("UNGROUNDED", kinds("Kilby is a 2.67 GW plant.", []))

    def test_wrong_year_from_real_source(self):
        self.assertIn("UNGROUNDED", kinds("Kilby first power is targeted for 2027.", ["ngv-factsheet"]))

    def test_campus_load_as_plant_capacity(self):
        self.assertIn("ENTITY_MIXUP", kinds("Kilby is a 2 GW plant.", ["news-cutrer"]))

    def test_minority_source_value(self):
        self.assertIn("CONTRADICTED", kinds("Kilby burns natural gas produced from Joulent's fields.", ["news-people"]))

    def test_superseded_decision_date(self):
        self.assertIn("STALE", kinds("The Kilby final investment decision is expected in 2026.", ["ngv-factsheet"]))

    def test_stale_is_cleared_when_answer_mentions_the_change(self):
        self.assertEqual(kinds("The Kilby final investment decision is expected in 2026. "
                               "Reporting says the final investment decision could slip to 2027.",
                               ["ngv-factsheet", "news-tnw"]), [])

    def test_static_fact_is_not_stale(self):  # bug found in eval round 1: capacity flagged STALE
        self.assertEqual(kinds("Kilby is a 2.67 GW plant.", ["ngv-factsheet"]), [])


class BrainAnswers(unittest.TestCase):
    def test_every_routed_answer_passes_its_own_checker(self):
        for q in ["What is Kilby's generation capacity?", "When is the Kilby final investment decision?",
                  "Who holds the Microsoft PPA?", "Whose gas fuels Kilby?", "How much will Kilby cost?"]:
            a = answer(q, LEDGER)
            self.assertEqual(check(a, LEDGER, DOCS), [], q)
            self.assertTrue(a.citations, q)

    def test_unknown_question_refuses(self):
        a = answer("What is the CEO's favorite color?", LEDGER)
        self.assertEqual(a.citations, [])
        self.assertIn("no answer rather than a guess", a.text)


if __name__ == "__main__":
    unittest.main()
