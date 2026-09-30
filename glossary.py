"""
Retrieval over a business glossary, for the model planner.

A table says "revenue" and a reader says "takings". A table has "region"
and a reader asks about "territories". The planner can only map words it
can see a reason for, so a reader who uploads a glossary of their own
terms gives it one. The glossary is split into definitions, the few most
relevant to a question are retrieved, and they are added to the planning
prompt, quoted as data.

Retrieval is lexical by default, with BM25 over the definitions. It needs
no model and no network, and it is what the tests measure. An embedding
function can be supplied as well; the two rankings are then combined by
reciprocal rank fusion, which needs no tuning of score scales between
them. Embeddings help when a question and a definition share a meaning
but no words.

What this is not: a way to answer a question from the glossary. The
retrieved text only helps the planner choose columns and filters. Every
figure is still computed from the table, and the plan is still validated
against it.
"""

import math
import re
from collections import Counter
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

# A definition longer than this is split, so one long entry cannot crowd
# the others out of the prompt.
MAX_CHUNK_CHARS = 600

# Definitions added to a planning prompt. A handful is enough to resolve
# the terms in one question.
DEFAULT_TOP_K = 3

# Reciprocal rank fusion constant. 60 is the value from the original paper
# and is not sensitive.
RRF_K = 60

# BM25 parameters, the usual defaults.
BM25_K1 = 1.5
BM25_B = 0.75

STOP_WORDS = frozenset({
    "a", "an", "and", "are", "as", "at", "be", "by", "did", "do", "does",
    "for", "from", "had", "has", "have", "how", "i", "in", "is", "it",
    "its", "me", "much", "of", "on", "or", "our", "show", "that", "the",
    "this", "to", "was", "we", "were", "what", "whats", "when", "which",
    "who", "with", "you",
})

Embedder = Callable[[Sequence[str]], list[list[float]]]


@dataclass(frozen=True)
class Chunk:
    """
    One definition, or part of one.

    Attributes:
        id:
            Stable position-based identifier, e.g. "g3".

        term:
            The heading or term the text defines, when there is one.

        text:
            The definition.
    """

    id: str
    term: str
    text: str

    def as_prompt_text(self) -> str:
        return f"{self.term}: {self.text}" if self.term else self.text


def tokenize(text: str) -> list[str]:
    """
    Lowercase words with a light plural fold, stop words removed.

    The fold is deliberately crude: "territories" and "territory" should
    meet, and a full stemmer would add a dependency for little gain on
    text this short.
    """

    words = re.findall(r"[a-z0-9]+", str(text).lower())
    folded = []

    for word in words:
        if word in STOP_WORDS:
            continue

        if len(word) > 4 and word.endswith("ies"):
            word = word[:-3] + "y"

        elif len(word) > 3 and word.endswith("s") and not word.endswith("ss"):
            word = word[:-1]

        folded.append(word)

    return folded


def _split_long(text: str) -> list[str]:
    if len(text) <= MAX_CHUNK_CHARS:
        return [text]

    sentences = re.split(r"(?<=[.!?])\s+", text)
    parts: list[str] = []
    current = ""

    for sentence in sentences:
        if current and len(current) + len(sentence) + 1 > MAX_CHUNK_CHARS:
            parts.append(current)
            current = sentence

        else:
            current = f"{current} {sentence}".strip()

    if current:
        parts.append(current)

    return parts


def chunk_glossary(text: str) -> list[Chunk]:
    """
    Split a glossary into definitions.

    Three layouts are read, because people write glossaries all three
    ways:

        Markdown headings, with the definition under each.
        "Term: definition" or "Term - definition" lines.
        Paragraphs separated by blank lines.

    Args:
        text:
            The glossary.

    Returns:
        The chunks, in document order.
    """

    entries: list[tuple[str, str]] = []
    term = ""
    body: list[str] = []

    def flush():
        joined = " ".join(line.strip() for line in body if line.strip())

        if joined or term:
            entries.append((term, joined))

    for line in str(text).splitlines():
        heading = re.match(r"^\s*#{1,6}\s+(.*\S)\s*$", line)
        definition = re.match(
            r"^\s*(?:[-*]\s+)?\*{0,2}([^:\n]{1,60}?)\*{0,2}\s*(?::|\s-\s|\s–\s)\s*(\S.*)$",
            line,
        )

        if heading:
            flush()
            term, body = heading.group(1).strip(), []

        elif definition and not body and not term:
            # Under a heading, a line is the heading's definition even
            # when it happens to contain a colon.
            flush()
            term, body = definition.group(1).strip(), [definition.group(2)]
            flush()
            term, body = "", []

        elif not line.strip():
            if body:
                flush()
                term, body = "", []

        else:
            body.append(line)

    flush()

    chunks = []

    for entry_term, entry_text in entries:
        if not entry_text:
            continue

        for part in _split_long(entry_text):
            chunks.append(Chunk(id=f"g{len(chunks) + 1}", term=entry_term, text=part))

    return chunks


class BM25:
    """Okapi BM25 over the chunks' terms and text."""

    def __init__(self, chunks: Sequence[Chunk]):
        self.documents = [
            tokenize(f"{chunk.term} {chunk.term} {chunk.text}") for chunk in chunks
        ]
        self.lengths = [len(document) for document in self.documents]
        self.average_length = (
            sum(self.lengths) / len(self.lengths) if self.lengths else 0.0
        )
        self.frequencies = [Counter(document) for document in self.documents]

        document_frequency: Counter = Counter()

        for document in self.documents:
            document_frequency.update(set(document))

        count = len(self.documents)
        self.idf = {
            word: math.log(1 + (count - seen + 0.5) / (seen + 0.5))
            for word, seen in document_frequency.items()
        }

    def scores(self, query: str) -> list[float]:
        words = tokenize(query)
        results = []

        for frequencies, length in zip(self.frequencies, self.lengths, strict=True):
            score = 0.0

            for word in words:
                if word not in frequencies:
                    continue

                tf = frequencies[word]
                norm = BM25_K1 * (
                    1 - BM25_B + BM25_B * length / (self.average_length or 1)
                )
                score += self.idf[word] * tf * (BM25_K1 + 1) / (tf + norm)

            results.append(score)

        return results


def _cosine(left: Sequence[float], right: Sequence[float]) -> float:
    dot = sum(a * b for a, b in zip(left, right, strict=True))
    norm = math.sqrt(sum(a * a for a in left)) * math.sqrt(sum(b * b for b in right))

    return dot / norm if norm else 0.0


@dataclass
class Glossary:
    """
    An indexed glossary.

    Attributes:
        chunks:
            The definitions.

        embedder:
            Optional function from texts to vectors. When set, rankings
            are fused with the lexical one.

        vectors:
            Chunk embeddings, computed once when the glossary is built.
    """

    chunks: list[Chunk]
    embedder: Embedder | None = None
    vectors: list[list[float]] = field(default_factory=list)

    def __post_init__(self):
        self.index = BM25(self.chunks)

        if self.embedder and self.chunks and not self.vectors:
            self.vectors = self.embedder([c.as_prompt_text() for c in self.chunks])

    @classmethod
    def from_text(cls, text: str, embedder: Embedder | None = None) -> "Glossary":
        return cls(chunk_glossary(text), embedder)

    def retrieve(self, question: str, k: int = DEFAULT_TOP_K) -> list[Chunk]:
        """
        The definitions most relevant to a question.

        A chunk needs at least one word in common with the question to be
        returned lexically. With an embedder, a chunk can also qualify on
        meaning alone, which is the point of having one.

        Args:
            question:
                What the reader typed.

            k:
                How many to return at most.

        Returns:
            Up to k chunks, best first.
        """

        if not self.chunks or k <= 0:
            return []

        lexical = self.index.scores(question)
        lexical_ranked = [
            position
            for position in sorted(
                range(len(self.chunks)), key=lambda p: lexical[p], reverse=True
            )
            if lexical[position] > 0
        ]

        rankings = [lexical_ranked]

        if self.embedder and self.vectors:
            query_vector = self.embedder([question])[0]
            similarity = [_cosine(query_vector, v) for v in self.vectors]
            rankings.append(sorted(
                range(len(self.chunks)), key=lambda p: similarity[p], reverse=True
            ))

        fused: dict[int, float] = {}

        for ranking in rankings:
            for rank, position in enumerate(ranking, start=1):
                fused[position] = fused.get(position, 0.0) + 1 / (RRF_K + rank)

        best = sorted(fused, key=lambda p: (-fused[p], p))[:k]

        return [self.chunks[position] for position in best]


def recall_at_k(
    glossary: Glossary,
    cases: Sequence[tuple[str, str]],
    k: int = DEFAULT_TOP_K
) -> float:
    """
    Share of questions whose expected term is among the top k retrieved.

    Args:
        glossary:
            The index.

        cases:
            (question, expected term) pairs.

        k:
            Depth.

    Returns:
        Recall in [0, 1].
    """

    if not cases:
        return 0.0

    hits = 0

    for question, term in cases:
        found = {chunk.term.lower() for chunk in glossary.retrieve(question, k)}
        hits += term.lower() in found

    return hits / len(cases)
