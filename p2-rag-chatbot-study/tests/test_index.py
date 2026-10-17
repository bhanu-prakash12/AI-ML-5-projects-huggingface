from study_assistant.index import add_chunks, clean_meta, search
from study_assistant.ingest import Chunk


class FakeEmb:
    """Tiny deterministic embeddings: no model, no network."""

    VOC = ["gradient", "tree", "kernel"]

    def _v(self, t: str) -> list[float]:
        t = t.lower()
        v = [float(t.count(w)) for w in self.VOC]
        return v if any(v) else [0.01, 0.01, 0.01]

    def embed_documents(self, texts):
        return [self._v(t) for t in texts]

    def embed_query(self, t):
        return self._v(t)


def make_store(tmp_path):
    from langchain_chroma import Chroma

    return Chroma(collection_name="test_col", embedding_function=FakeEmb(), persist_directory=str(tmp_path),
                  collection_metadata={"hnsw:space": "cosine"})


def chunks():
    return [
        Chunk("gradient descent follows the gradient", {"source": "a.md", "doc_type": "notes", "page": None}),
        Chunk("a decision tree splits data", {"source": "b.md", "doc_type": "notes"}),
        Chunk("kernel trick question", {"source": "e.pdf", "doc_type": "exam", "page": 2}),
    ]


def test_clean_meta_drops_none():
    assert clean_meta({"a": 1, "b": None}) == {"a": 1}


def test_add_is_idempotent(tmp_path):
    s = make_store(tmp_path)
    assert add_chunks(s, chunks()) == 3
    assert add_chunks(s, chunks()) == 0
    assert s._collection.count() == 3


def test_search_and_filter(tmp_path):
    s = make_store(tmp_path)
    add_chunks(s, chunks())
    top = search(s, "gradient", k=1)
    assert top[0][0].metadata["source"] == "a.md" and top[0][1] > 0.9
    ex = search(s, "gradient", k=3, doc_type="exam")
    assert [d.metadata["doc_type"] for d, _ in ex] == ["exam"]
