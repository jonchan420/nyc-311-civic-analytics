"""Retrieval over the method notes in docs/knowledge/.

Numbers in answers always come from tools; retrieval supplies the method,
definitions, and limits that go with them. With about 30 short sections,
a small BM25 keyword index is enough, and every score can be explained.
Retrieved text is data for the writer to quote, never instructions.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from config.settings import PROJECT_ROOT

KNOWLEDGE_DIR = PROJECT_ROOT / "docs" / "knowledge"
STOPWORDS = set("""a an and are as at be by can do does for from has have how i if in into is it its
of on or so than that the their them then there these this to was were what when where which
who why will with you your""".split())


@dataclass(frozen=True)
class Chunk:
    doc: str      # file name without .md
    title: str    # the document's "# " heading
    section: str  # the "## " heading
    text: str

    @property
    def id(self) -> str:
        return f"{self.doc}#{re.sub(r'[^a-z0-9]+', '-', self.section.lower()).strip('-')}"


def tokenize(text: str) -> list[str]:
    tokens = re.findall(r"[a-z0-9]+", text.lower())
    # light stemming: "complaints" and "complaint" should match; one-letter
    # tokens are contraction fragments ("can't" -> "can", "t") and are dropped
    return [t[:-1] if len(t) > 3 and t.endswith("s") and not t.endswith("ss") else t
            for t in tokens if len(t) > 1 and t not in STOPWORDS]


def load_chunks(knowledge_dir: Path = KNOWLEDGE_DIR) -> list[Chunk]:
    chunks = []
    for path in sorted(Path(knowledge_dir).glob("*.md")):
        text = path.read_text()
        title_match = re.search(r"^# (.+)$", text, flags=re.MULTILINE)
        title = title_match.group(1).strip() if title_match else path.stem
        for match in re.finditer(r"^## (.+?)\n(.*?)(?=^## |\Z)", text, flags=re.MULTILINE | re.DOTALL):
            chunks.append(Chunk(path.stem, title, match.group(1).strip(), match.group(2).strip()))
    return chunks


class KnowledgeBase:
    """BM25 (k1=1.5, b=0.75) over title + section heading + text of each chunk."""

    def __init__(self, chunks: list[Chunk], k1: float = 1.5, b: float = 0.75):
        self.chunks = chunks
        self.k1, self.b = k1, b
        self._terms = [Counter(tokenize(f"{c.title} {c.section} {c.text}")) for c in chunks]
        self._lengths = [sum(terms.values()) for terms in self._terms]
        self._avg_length = sum(self._lengths) / max(len(chunks), 1)
        doc_freq = Counter(term for terms in self._terms for term in terms)
        n = len(chunks)
        self._idf = {term: math.log(1 + (n - df + 0.5) / (df + 0.5)) for term, df in doc_freq.items()}

    @classmethod
    def load(cls, knowledge_dir: Path = KNOWLEDGE_DIR) -> KnowledgeBase:
        return cls(load_chunks(knowledge_dir))

    def scores(self, query: str) -> list[float]:
        query_terms = set(tokenize(query))
        out = []
        for terms, length in zip(self._terms, self._lengths):
            score = 0.0
            for term in query_terms & terms.keys():
                tf = terms[term]
                score += self._idf[term] * tf * (self.k1 + 1) / (
                    tf + self.k1 * (1 - self.b + self.b * length / self._avg_length))
            out.append(score)
        return out

    def search(self, query: str, k: int = 3) -> list[tuple[Chunk, float]]:
        ranked = sorted(zip(self.chunks, self.scores(query)), key=lambda pair: (-pair[1], pair[0].id))
        return [(chunk, round(score, 3)) for chunk, score in ranked[:k] if score > 0]
