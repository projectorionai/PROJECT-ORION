"""Routing caches must retain exact inputs, model generation and bounded memory."""
from concurrent.futures import ThreadPoolExecutor
import gc
import threading
import weakref
import numpy as np
import pytest

from orion_core import semantic, tool_resolver as tr


@pytest.fixture(autouse=True)
def private_caches(monkeypatch):
    for name in ("_TOOL_VECTORS", "_TOOL_MATRIX", "_EXAMPLE_VECTORS", "_EXAMPLE_MATRIX",
                 "_INDEX_CACHE"):
        monkeypatch.setattr(tr, name, getattr(tr, "_RoutingCache", dict)(), raising=False)
    monkeypatch.setattr(tr, "_INDEX", None)
    monkeypatch.setattr(tr, "vocabulary_for", lambda _name: "")


def declarations(word="copper"):
    return [{"name": "remote_inventory", "description": f"Read the {word} inventory."},
            {"name": "unrelated", "description": "Inspect the weather."}]


@pytest.mark.parametrize("inplace", [False, True])
def test_same_named_tool_description_change_replaces_stale_ranking(inplace):
    first = declarations()
    assert tr.lexical_scores("copper", first)["remote_inventory"] > 0
    changed = first if inplace else declarations()
    changed[0]["description"] = "Read the zirconium inventory."
    assert tr.lexical_scores("zirconium", changed)["remote_inventory"] > 0
    assert tr.lexical_scores("copper", changed)["remote_inventory"] == 0


def test_vocabulary_update_is_not_hidden_by_the_same_tool_names(monkeypatch):
    words = {"remote_inventory": "copper"}
    monkeypatch.setattr(tr, "vocabulary_for", lambda name: words.get(name, ""))
    tools = declarations("ordinary")
    assert tr.lexical_scores("copper", tools)["remote_inventory"] > 0
    words["remote_inventory"] = "zirconium"
    assert tr.lexical_scores("zirconium", tools)["remote_inventory"] > 0
    assert tr.lexical_scores("copper", tools)["remote_inventory"] == 0


def test_alternating_full_and_gateway_menus_build_each_index_once(monkeypatch):
    original, built = tr._LexicalIndex, []

    def counted(*args, **kwargs):
        built.append(1)
        return original(*args, **kwargs)

    monkeypatch.setattr(tr, "_LexicalIndex", counted)
    full = declarations() + [{"name": "find_tool"}, {"name": "use_tool"}]
    gateway = declarations()
    for _ in range(8):
        for menu in (full, gateway):
            assert tr.lexical_scores("copper", menu)["remote_inventory"] > 0
    assert len(built) == 2


def test_real_text_and_voice_menus_fit_the_bounded_index_cache(monkeypatch):
    from orion_core.dispatch_schema import TOOL_DECLARATIONS
    from orion_core.tool_vocabulary import terms_for
    monkeypatch.setattr(tr, "vocabulary_for", terms_for)
    original, built = tr._LexicalIndex, []

    def counted(*args, **kwargs):
        built.append(1)
        return original(*args, **kwargs)

    monkeypatch.setattr(tr, "_LexicalIndex", counted)
    gateway = [item for item in TOOL_DECLARATIONS if item["name"] not in {"find_tool", "use_tool"}]
    for _ in range(3):
        for menu in (TOOL_DECLARATIONS, gateway):
            assert tr.lexical_scores("calculate two plus two", menu)["calculate"] > 0
    assert len(built) == 2


class Encoder:
    def __init__(self, dimension=4):
        self.dimension = dimension
        self._session = object()

    def ready(self):
        return True

    def encode(self, texts):
        return np.vstack([np.full(self.dimension, len(text) or 1, dtype=np.float32)
                          for text in texts])

    def encode_one(self, text):
        return self.encode([text])[0]


def test_alternating_semantic_matrices_retain_both_schemas(monkeypatch):
    monkeypatch.setattr(semantic, "ENCODER", Encoder())
    first = tr._tool_matrix(declarations())[1]
    tr._tool_matrix(declarations()[:1])
    assert tr._tool_matrix(declarations())[1] is first


@pytest.mark.parametrize("change", ["encoder", "session"])
def test_new_encoder_or_session_does_not_reuse_old_dimension_vectors(monkeypatch, change):
    encoder = Encoder(4)
    monkeypatch.setattr(semantic, "ENCODER", encoder)
    assert tr._tool_matrix(declarations())[1].shape == (2, 4)
    if change == "encoder":
        monkeypatch.setattr(semantic, "ENCODER", Encoder(3))
    else:
        encoder.dimension = 3
        encoder._session = object()
    assert tr._tool_matrix(declarations())[1].shape == (2, 3)


@pytest.mark.parametrize("entrypoint", ["semantic_scores", "blank_scores", "hybrid_scores",
                                       "semantic_ready", "warm_semantic"])
def test_cold_replacement_releases_previous_encoder_and_vectors(monkeypatch, entrypoint):
    from orion_core import spelling

    class Cold(Encoder):
        def ready(self):
            return False

        def encode(self, texts):
            raise AssertionError("A cold encoder must not be used for inference")

    monkeypatch.setattr(spelling, "correct", lambda text: text)
    monkeypatch.setattr(spelling, "speller", lambda: None)
    old = Encoder()
    monkeypatch.setattr(semantic, "ENCODER", old)
    tr._tool_matrix(declarations())
    monkeypatch.setattr(tr, "_tool_examples", lambda _tools: [["old request"], []])
    tr._example_matrix(declarations())
    assert tr._TOOL_VECTORS and tr._EXAMPLE_VECTORS
    previous = weakref.ref(old)
    # Assign directly: a second monkeypatch would itself retain the old encoder.
    semantic.ENCODER = Cold()
    del old

    if entrypoint == "semantic_ready":
        assert tr.semantic_ready() is False
    elif entrypoint == "warm_semantic":
        assert tr.warm_semantic(declarations()) == 0
    elif entrypoint == "hybrid_scores":
        assert tr.hybrid_scores("copper", declarations()) == tr.lexical_scores("copper", declarations())
    else:
        assert tr.semantic_scores("" if entrypoint == "blank_scores" else "copper", declarations()) == {}

    gc.collect()
    assert previous() is None
    assert not any((tr._TOOL_VECTORS, tr._TOOL_MATRIX, tr._EXAMPLE_VECTORS, tr._EXAMPLE_MATRIX))


def test_same_named_description_change_rebuilds_semantic_documents(monkeypatch):
    monkeypatch.setattr(semantic, "ENCODER", Encoder())
    before = tr._tool_matrix(declarations("copper"))[1].copy()
    after = tr._tool_matrix(declarations("muchlongerzirconium"))[1]
    assert not np.array_equal(before, after)


def test_late_old_encoder_worker_cannot_poison_current_vectors(monkeypatch):
    entered, release = threading.Event(), threading.Event()

    class Old(Encoder):
        def encode(self, texts):
            entered.set()
            assert release.wait(5)
            return super().encode(texts)

    monkeypatch.setattr(semantic, "ENCODER", Old(4))
    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(tr._tool_matrix, declarations())
        try:
            assert entered.wait(3)
            monkeypatch.setattr(semantic, "ENCODER", Encoder(3))
            assert tr._tool_matrix(declarations())[1].shape == (2, 3)
        finally:
            release.set()
        future.result(timeout=3)
    assert tr._tool_matrix(declarations())[1].shape == (2, 3)


def test_oversized_index_is_used_without_being_retained(monkeypatch):
    monkeypatch.setattr(tr, "_LEXICAL_CACHE_BYTES", 1, raising=False)
    assert tr.lexical_scores("copper", declarations())["remote_inventory"] > 0
    assert not tr._INDEX_CACHE and tr._INDEX is None


def test_semantic_budget_does_not_drop_uncached_tools_or_change_scores(monkeypatch):
    monkeypatch.setattr(semantic, "ENCODER", Encoder(256))
    monkeypatch.setattr(tr, "_VECTOR_CACHE_BYTES", 1, raising=False)
    monkeypatch.setattr(tr, "_MATRIX_CACHE_BYTES", 1, raising=False)
    names, matrix = tr._tool_matrix(declarations())
    assert names == ["remote_inventory", "unrelated"] and matrix.shape == (2, 256)
    assert not tr._TOOL_VECTORS and not tr._TOOL_MATRIX


def test_schema_caches_evict_instead_of_growing_with_plugin_updates(monkeypatch):
    monkeypatch.setattr(semantic, "ENCODER", Encoder())
    for index in range(12):
        tools = declarations(f"synthetic{index}")
        tr.lexical_scores(f"synthetic{index}", tools)
        tr._tool_matrix(tools)
    assert len(tr._INDEX_CACHE) <= 4 and len(tr._TOOL_MATRIX) <= 4


def test_retained_vector_rows_do_not_pin_the_encoders_entire_batch(monkeypatch):
    monkeypatch.setattr(semantic, "ENCODER", Encoder(256))
    tr._tool_matrix(declarations())
    assert tr._TOOL_VECTORS
    assert all(row.base is None and not row.flags.writeable for row in tr._TOOL_VECTORS.values())


def test_reset_discards_semantic_example_matrices_when_curated_examples_change(monkeypatch):
    monkeypatch.setattr(semantic, "ENCODER", Encoder())
    examples = [["old request"], []]
    monkeypatch.setattr(tr, "_tool_examples", lambda _tools: examples)
    first = tr._example_matrix(declarations())[2].copy()
    examples[0] = ["a different much longer synthetic request"]
    tr.reset_index()
    assert not np.array_equal(tr._example_matrix(declarations())[2], first)


class PinnedEncoder(Encoder):
    """Like the real encoder: a named, pinned model with a known dimension."""

    def __init__(self, dimension=4):
        super().__init__(dimension)
        from types import SimpleNamespace
        self.spec = SimpleNamespace(name="fake-model", dim=dimension,
                                    files={"m": SimpleNamespace(sha256="ab" * 32)})
        self.encoded: list[str] = []

    def encode(self, texts):
        self.encoded.extend(texts)
        return super().encode(texts)


def _fresh_memory_caches(monkeypatch):
    for name in ("_TOOL_VECTORS", "_TOOL_MATRIX", "_EXAMPLE_VECTORS", "_EXAMPLE_MATRIX"):
        monkeypatch.setattr(tr, name, tr._RoutingCache())
    monkeypatch.setattr(tr, "_DISK", None)


def test_tool_vectors_survive_a_restart_without_re_encoding(monkeypatch, tmp_path):
    # 23 Oct: every launch re-embedded ~1,000 routing texts (4.7 s of the
    # semantic-warm thread). A restart must read them back instead.
    monkeypatch.setattr(tr, "VECTOR_CACHE_PATH", tmp_path / "routing_vectors.npz")
    _fresh_memory_caches(monkeypatch)
    first = PinnedEncoder()
    monkeypatch.setattr(semantic, "ENCODER", first)
    names, matrix = tr._tool_matrix(declarations())
    assert len(first.encoded) == 2 and (tmp_path / "routing_vectors.npz").is_file()

    _fresh_memory_caches(monkeypatch)            # a new process
    second = PinnedEncoder()
    monkeypatch.setattr(semantic, "ENCODER", second)
    again_names, again = tr._tool_matrix(declarations())
    assert second.encoded == []
    assert again_names == names and np.array_equal(again, matrix)

    _fresh_memory_caches(monkeypatch)            # one description edited
    third = PinnedEncoder()
    monkeypatch.setattr(semantic, "ENCODER", third)
    tr._tool_matrix(declarations("tin"))
    assert len(third.encoded) == 1 and "tin" in third.encoded[0]


def test_stored_vectors_are_never_reused_by_another_model_or_a_fake(monkeypatch, tmp_path):
    monkeypatch.setattr(tr, "VECTOR_CACHE_PATH", tmp_path / "routing_vectors.npz")
    _fresh_memory_caches(monkeypatch)
    monkeypatch.setattr(semantic, "ENCODER", PinnedEncoder(4))
    tr._tool_matrix(declarations())

    _fresh_memory_caches(monkeypatch)
    other = PinnedEncoder(3)                      # a different model dimension
    monkeypatch.setattr(semantic, "ENCODER", other)
    assert tr._tool_matrix(declarations())[1].shape == (2, 3) and len(other.encoded) == 2

    _fresh_memory_caches(monkeypatch)
    plain = Encoder(4)                            # no model identity at all
    monkeypatch.setattr(semantic, "ENCODER", plain)
    assert tr._disk_lookup(plain, ["remote inventory."]) == {}


def test_nothing_is_written_under_pytest_without_an_explicit_path(monkeypatch):
    monkeypatch.setattr(tr, "VECTOR_CACHE_PATH", None)
    assert tr._vector_cache_path() is None
