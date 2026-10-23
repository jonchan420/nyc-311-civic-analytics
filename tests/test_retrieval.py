"""BM25 retrieval over docs/knowledge/.

DEV_QUESTIONS were used while writing the notes (two notes were reworded
to fix misses), so they measure fit, not generalization. The held-out
retrieval questions live in eval/ and aren't used for tuning.
"""

import math

import pytest

from analyst.retrieval import Chunk, KnowledgeBase, load_chunks, tokenize

DEV_QUESTIONS = [
    ("Why don't borough totals add up to the citywide total?", "boroughs-and-per-capita"),
    ("What does a renamed flag mean?", "categories-and-renames"),
    ("Why was Sewer replaced by Sewer Maintenance?", "categories-and-renames"),
    ("Why can't I see this month's numbers yet?", "complete-months"),
    ("Does a drop in complaints mean conditions improved?", "recorded-vs-actual"),
    ("How is forecast accuracy scored?", "forecasts"),
    ("What is WAPE?", "forecasts"),
    ("Why did the forecast come in too low?", "forecasts"),
    ("What does it mean when a month is flagged as unusual?", "unusual-months"),
    ("Can I get response times for my neighborhood?", "scope"),
    ("How are per capita rates calculated?", "boroughs-and-per-capita"),
    ("Which months count as summer?", "scope"),
    ("Do Chinese-speaking neighborhoods report less?", "recorded-vs-actual"),
    ("Where does the data come from?", "data-and-grain"),
    ("What does one row in the table mean?", "data-and-grain"),
]


@pytest.fixture(scope="module")
def kb():
    return KnowledgeBase.load()


def test_every_note_has_a_title_and_sections():
    chunks = load_chunks()
    assert {c.doc for c in chunks} == {
        "boroughs-and-per-capita", "categories-and-renames", "complete-months", "data-and-grain",
        "forecasts", "recorded-vs-actual", "scope", "unusual-months"}
    assert all(c.title and c.section and c.text for c in chunks)
    assert len({c.id for c in chunks}) == len(chunks)


@pytest.mark.parametrize("question, doc", DEV_QUESTIONS)
def test_dev_question_finds_its_note_in_the_top_3(kb, question, doc):
    assert doc in [chunk.doc for chunk, _ in kb.search(question, k=3)]


def test_tokenize_drops_stopwords_and_plural_s():
    assert tokenize("What are the complaints in Queens?") == ["complaint", "queen"]
    assert tokenize("Address class") == ["address", "class"]   # "ss" endings are kept


def test_bm25_scores_match_a_hand_calculation():
    chunks = [Chunk("a", "Notes", "Part", "heat heat noise"), Chunk("b", "Notes", "Part", "noise parking"),
              Chunk("c", "Notes", "Part", "parking parking parking")]
    kb = KnowledgeBase(chunks)
    # each chunk also has "note" and "part" from its title and section: lengths 5, 4, 5; average 14/3
    # idf(heat) = ln(1 + (3 - 1 + 0.5) / (1 + 0.5)); tf = 2 in chunk a
    idf = math.log(1 + 2.5 / 1.5)
    expected = idf * 2 * 2.5 / (2 + 1.5 * (1 - 0.75 + 0.75 * 5 / (14 / 3)))
    assert kb.scores("heat")[0] == pytest.approx(expected)
    assert kb.scores("heat")[1:] == [0.0, 0.0]
    assert [c.doc for c, _ in kb.search("parking")] == ["c", "b"]   # more occurrences rank higher


def test_no_matching_words_returns_nothing(kb):
    assert kb.search("zzzz qqqq") == []
