from study_assistant.ingest import (
    Chunk, chunk_exam_pages, chunk_notes, ingest_corpus, parse_front_matter, split_questions,
)

EXAM = """Machine Learning Final 2021
Answer all questions.
Question 1. (10 marks) Define overfitting.
(a) Give an example.
Question 2. (15 marks) Explain gradient descent.
Question 3. (5 marks) What is regularization?
"""
EXAM_NUM = "1. Define bias.\n2. Define variance.\n3) Define noise.\n"
NOTE = "---\ntitle: Overfitting\nurl: https://en.wikipedia.org/wiki/Overfitting\nrevision: 42\n---\n# Overfitting\n\n" + (
    "Overfitting is when a model fits noise. " * 60
)


def test_split_questions_question_style():
    qs = split_questions(EXAM)
    assert [q[0] for q in qs] == ["1", "2", "3"]
    assert "(a) Give an example." in qs[0][1]


def test_split_questions_numeric_style():
    assert [q[0] for q in split_questions(EXAM_NUM)] == ["1", "2", "3"]


def test_split_questions_no_markers():
    assert split_questions("just prose without any numbering at all") == []


def test_exam_fallback_rate():
    chunks, stats = chunk_exam_pages([EXAM, "plain prose page with no questions " * 5], "exam_2021.pdf", 2021)
    assert stats["question_chunks"] == 3 and stats["fallback_chunks"] >= 1
    assert chunks[0].metadata["doc_type"] == "exam" and chunks[0].metadata["page"] == 1
    assert chunks[0].metadata["question"] == "1"


def test_notes_front_matter_and_chunks():
    meta, body = parse_front_matter(NOTE)
    assert meta["revision"] == "42" and body.startswith("# Overfitting")
    chunks = chunk_notes(NOTE, "overfitting.md")
    assert len(chunks) > 1 and all(len(c.text) <= 800 for c in chunks)
    assert chunks[0].metadata["url"].endswith("Overfitting")


def test_ids_stable_and_content_sensitive():
    a, b = chunk_notes(NOTE, "o.md"), chunk_notes(NOTE, "o.md")
    assert [c.id for c in a] == [c.id for c in b]
    assert Chunk("x", {"source": "s", "page": 1}).id != Chunk("y", {"source": "s", "page": 1}).id


def test_ingest_idempotent(tmp_path):
    (tmp_path / "notes").mkdir()
    (tmp_path / "notes" / "o.md").write_text(NOTE)
    c1, s1 = ingest_corpus(tmp_path)
    c2, _ = ingest_corpus(tmp_path)
    assert {c.id for c in c1} == {c.id for c in c2}
    assert s1["notes_docs"] == 1 and s1["total_chunks"] == len(c1)


def test_split_questions_problem_style():
    text = "Problem 1\nConsider AdaBoost.\nProblem 2\nExplain SVM margins.\n"
    assert [q[0] for q in split_questions(text)] == ["1", "2"]
