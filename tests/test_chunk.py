from tts_tool.chunk import Chunk, normalize, pack, split_paragraphs


def test_normalize_strips_and_collapses_crlf():
    assert normalize("  hello\r\nworld  ") == "hello\nworld"
    assert normalize("\r\nfoo\r\n") == "foo"


def test_split_paragraphs_drops_empty():
    assert split_paragraphs("a\n\nb\n\n\n\nc") == ["a", "b", "c"]
    assert split_paragraphs("") == []


def test_pack_empty():
    assert pack([]) == []


def test_pack_single_sentence_under_target():
    chunks = pack([("Hello world.", 0)])
    assert chunks == [Chunk(text="Hello world.", index=0)]


def test_pack_never_splits_a_sentence():
    long = "x" * 3000 + "."
    chunks = pack([(long, 0)])
    assert len(chunks) == 1
    assert chunks[0].text == long


def test_pack_respects_target_across_sentences():
    target = 100
    sentences = [(f"Sentence number {i} fills space.", 0) for i in range(20)]
    chunks = pack(sentences, target=target, soft=60)
    assert len(chunks) >= 2
    for c in chunks:
        if c.text.count(" ") > 0:
            assert len(c.text) <= target or " " not in c.text


def test_pack_soft_closes_at_paragraph_boundary_when_above_soft():
    sentences = [
        ("a" * 80 + ".", 0),
        ("b" * 80 + ".", 1),
    ]
    chunks = pack(sentences, target=2000, soft=50)
    assert len(chunks) == 2
    assert chunks[0].text.startswith("aaaa")
    assert chunks[1].text.startswith("bbbb")


def test_pack_does_not_soft_close_below_threshold():
    sentences = [("short.", 0), ("also short.", 1)]
    chunks = pack(sentences, target=2000, soft=100)
    assert len(chunks) == 1
    assert "\n\n" in chunks[0].text


def test_render_preserves_paragraph_breaks_within_chunk():
    sentences = [("First.", 0), ("Second.", 0), ("New para.", 1)]
    chunks = pack(sentences, target=2000, soft=2000)
    assert len(chunks) == 1
    assert chunks[0].text == "First. Second.\n\nNew para."
