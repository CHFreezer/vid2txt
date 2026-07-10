"""Unit tests for src.formatter."""

from src.formatter import Formatter


class TestToTxt:
    def test_empty_segments(self) -> None:
        fmt = Formatter()
        assert fmt.to_txt([]) == ""

    def test_single_segment(self) -> None:
        fmt = Formatter()
        segments = [{"start": 0.0, "end": 2.5, "text": "Hello world", "translated_text": None}]
        assert fmt.to_txt(segments) == "Hello world"

    def test_multiple_segments(self) -> None:
        fmt = Formatter()
        segments = [
            {"start": 0.0, "end": 2.0, "text": "First line", "translated_text": None},
            {"start": 2.5, "end": 5.0, "text": "Second line", "translated_text": None},
        ]
        assert fmt.to_txt(segments) == "First line\nSecond line"

    def test_joins_with_newline(self) -> None:
        fmt = Formatter()
        segments = [
            {"start": 0.0, "end": 1.0, "text": "A", "translated_text": None},
            {"start": 1.0, "end": 2.0, "text": "B", "translated_text": None},
            {"start": 2.0, "end": 3.0, "text": "C", "translated_text": None},
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
            {"start": 0.0, "end": 2.0, "text": "One", "translated_text": None},
            {"start": 2.0, "end": 4.0, "text": "Two", "translated_text": None},
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


class TestToTranslatedTxt:
    def test_empty(self) -> None:
        fmt = Formatter()
        assert fmt.to_translated_txt([]) == ""

    def test_single(self) -> None:
        fmt = Formatter()
        segs = [{"start": 0.0, "end": 2.0, "text": "你好", "translated_text": "Hello"}]
        assert fmt.to_translated_txt(segs) == "Hello"

    def test_skips_none_translation(self) -> None:
        fmt = Formatter()
        segs = [{"start": 0.0, "end": 1.0, "text": "A", "translated_text": None}]
        assert fmt.to_translated_txt(segs) == ""


class TestToTranslatedSrt:
    def test_single(self) -> None:
        fmt = Formatter()
        segs = [{"start": 0.0, "end": 1.0, "text": "你好", "translated_text": "Hello"}]
        result = fmt.to_translated_srt(segs)
        assert "Hello" in result
        assert "你好" not in result


class TestToBilingualTxt:
    def test_contains_both_languages(self) -> None:
        fmt = Formatter()
        segs = [{"start": 0.0, "end": 1.0, "text": "你好", "translated_text": "Hello"}]
        result = fmt.to_bilingual_txt(segs)
        assert "你好" in result
        assert "Hello" in result
        assert "🎙" in result
        assert "🌐" in result


class TestToBilingualSrt:
    def test_contains_both_languages(self) -> None:
        fmt = Formatter()
        segs = [{"start": 0.0, "end": 1.0, "text": "你好", "translated_text": "Hello"}]
        result = fmt.to_bilingual_srt(segs)
        assert "你好" in result
        assert "Hello" in result


class TestWriteWithTranslation:
    def test_writes_all_files_when_translated(self, tmp_path) -> None:
        import os as _os
        fmt = Formatter()
        segs = [{"start": 0.0, "end": 1.0, "text": "你好", "translated_text": "Hello"}]
        out = fmt.write(segs, str(tmp_path), "test", translated=True, target_lang="en")
        assert _os.path.isfile(out["txt"])
        assert _os.path.isfile(out["srt"])
        assert _os.path.isfile(out["translated_txt"])
        assert _os.path.isfile(out["translated_srt"])
        assert _os.path.isfile(out["bilingual_txt"])

    def test_no_translated_files_when_not_translated(self, tmp_path) -> None:
        fmt = Formatter()
        segs = [{"start": 0.0, "end": 1.0, "text": "你好", "translated_text": None}]
        out = fmt.write(segs, str(tmp_path), "test", translated=False)
        assert "translated_txt" not in out
        assert "translated_srt" not in out
        assert "bilingual_txt" not in out
