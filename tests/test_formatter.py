"""Unit tests for src.formatter."""

from src.formatter import Formatter


class TestToTxt:
    def test_empty_segments(self) -> None:
        fmt = Formatter()
        assert fmt.to_txt([]) == ""

    def test_single_segment(self) -> None:
        fmt = Formatter()
        segments = [{"start": 0.0, "end": 2.5, "text": "Hello world"}]
        assert fmt.to_txt(segments) == "Hello world"

    def test_multiple_segments(self) -> None:
        fmt = Formatter()
        segments = [
            {"start": 0.0, "end": 2.0, "text": "First line"},
            {"start": 2.5, "end": 5.0, "text": "Second line"},
        ]
        assert fmt.to_txt(segments) == "First line\nSecond line"

    def test_joins_with_newline(self) -> None:
        fmt = Formatter()
        segments = [
            {"start": 0.0, "end": 1.0, "text": "A"},
            {"start": 1.0, "end": 2.0, "text": "B"},
            {"start": 2.0, "end": 3.0, "text": "C"},
        ]
        result = fmt.to_txt(segments)
        assert result.count("\n") == 2


class TestToSrt:
    def test_empty_segments(self) -> None:
        fmt = Formatter()
        assert fmt.to_srt([]) == ""

    def test_single_segment(self) -> None:
        fmt = Formatter()
        segments = [{"start": 1.5, "end": 3.0, "text": "Hello"}]
        expected = "1\n00:00:01,500 --> 00:00:03,000\nHello\n"
        assert fmt.to_srt(segments) == expected

    def test_multiple_segments_sequential_numbers(self) -> None:
        fmt = Formatter()
        segments = [
            {"start": 0.0, "end": 2.0, "text": "One"},
            {"start": 2.0, "end": 4.0, "text": "Two"},
        ]
        result = fmt.to_srt(segments)
        lines = result.split("\n")
        # First entry
        assert lines[0] == "1"
        assert "00:00:00,000 --> 00:00:02,000" in lines[1]
        assert lines[2] == "One"
        # Blank separator
        assert lines[3] == ""
        # Second entry
        assert lines[4] == "2"
        assert "00:00:02,000 --> 00:00:04,000" in lines[5]
        assert lines[6] == "Two"

    def test_ends_with_newline(self) -> None:
        fmt = Formatter()
        segments = [{"start": 0.0, "end": 1.0, "text": "Test"}]
        result = fmt.to_srt(segments)
        assert result.endswith("\n")
