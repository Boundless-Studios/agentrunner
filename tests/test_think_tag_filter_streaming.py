"""Unit tests for ThinkTagFilter — the streaming counterpart to ThinkTagStrippingModel.

Covers:
- Paired ``<think>...</think>`` stripping across chunk boundaries.
- Orphan ``</think>`` (server-prefilled reasoning preamble — Baseten/Nemotron
  pattern observed in Langfuse trace ``51298337a65a4521967e3b3215d87700``).
- Pass-through after real content has been yielded.
"""
from agentrunner import ThinkTagFilter


def _feed_all(filt: ThinkTagFilter, chunks: list[str]) -> str:
    """Feed a sequence of chunks and return concatenated output."""
    return "".join(filt.feed(c) for c in chunks)


class TestThinkTagFilterPaired:
    """Existing paired-tag behavior — regression coverage."""

    def test_clean_text_passes_through(self):
        f = ThinkTagFilter()
        assert f.feed("hello world") == "hello world"

    def test_paired_block_stripped_in_one_chunk(self):
        f = ThinkTagFilter()
        result = f.feed("<think>reasoning</think>final")
        assert result == "final"

    def test_paired_block_split_across_chunks(self):
        f = ThinkTagFilter()
        result = _feed_all(f, ["<think>partial reasoning", " more reasoning</think>", "final"])
        assert result == "final"

    def test_close_tag_split_across_chunks(self):
        f = ThinkTagFilter()
        result = _feed_all(f, ["<think>reasoning</thi", "nk>final answer"])
        assert result == "final answer"


class TestThinkTagFilterOrphanClose:
    """BOU-818: orphan ``</think>`` from server-prefilled reasoning mode.

    Baseten's Nemotron deployment prefills the assistant turn with
    ``<think>``, so we only see the closing tag on the wire. Drop everything
    before+including the first ``</think>`` until real content has been
    yielded.
    """

    def test_orphan_close_in_first_chunk_strips_preamble(self):
        f = ThinkTagFilter()
        result = f.feed("reasoning preamble</think>actual answer")
        assert result == "actual answer"

    def test_orphan_close_with_whitespace_matches_production(self):
        """Exact shape from Langfuse trace 51298337a65a4521967e3b3215d87700."""
        f = ThinkTagFilter()
        text = (
            "\nMood: Frantic defiance\n"
            "Stakes: lock the party out\n"
            "Primary speakers: None\n\n"
            "</think>\n\n"
            "Mood: Frantic defiance\n"
            "Stakes: lock the party out\n"
            "Primary speakers: None"
        )
        result = f.feed(text)
        assert "</think>" not in result
        # Only the final (post-close) copy survives.
        assert result.count("Mood: Frantic defiance") == 1
        assert "Mood: Frantic defiance" in result

    def test_orphan_close_split_across_chunks(self):
        f = ThinkTagFilter()
        result = _feed_all(
            f,
            ["draft preamble</thi", "nk>", "final"],
        )
        assert result == "final"

    def test_orphan_close_after_real_content_passes_through(self):
        """Once we've yielded real content, an orphan ``</think>`` is treated
        as literal text — the preamble window has closed.
        """
        f = ThinkTagFilter()
        first = f.feed("real content already streamed ")
        second = f.feed("now a stray </think> appears mid-stream")
        assert "real content" in first
        assert "</think>" in second  # passed through

    def test_orphan_close_then_more_chunks(self):
        """Streaming continues normally after the orphan close."""
        f = ThinkTagFilter()
        result = _feed_all(
            f,
            ["draft preamble</think>chunk one ", "chunk two ", "chunk three"],
        )
        assert result == "chunk one chunk two chunk three"

    def test_strip_static_method_handles_orphan_close(self):
        """The one-shot ``strip()`` mirrors streaming behavior."""
        text = "preamble</think>final"
        assert ThinkTagFilter.strip(text) == "final"

    def test_strip_static_method_paired_then_orphan(self):
        text = "<think>nested</think>draft</think>final"
        assert ThinkTagFilter.strip(text) == "final"
