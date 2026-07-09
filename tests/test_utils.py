"""Unit tests for src.utils."""

import pytest
from src.utils import (
    validate_url,
    format_timestamp,
    get_output_basename,
    check_dependencies,
)


class TestValidateURL:
    @pytest.mark.parametrize("url", [
        "https://www.bilibili.com/video/BV1GJ41177UQ",
        "https://bilibili.com/video/BV1xx41c7EQ",
        "https://b23.tv/xxxxxx",
        "https://www.b23.tv/abcdef",
        "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
        "https://youtube.com/watch?v=abcdefghijk",
        "https://www.youtube.com/shorts/abc123def45",
        "https://youtube.com/shorts/abc123def45",
        "https://youtu.be/dQw4w9WgXcQ",
    ])
    def test_valid_urls(self, url: str) -> None:
        assert validate_url(url) is True

    @pytest.mark.parametrize("url", [
        "",
        "not a url",
        "https://www.example.com/video/123",
        "https://bilibili.com/",
        "https://www.bilibili.com/",
        "https://youtube.com/",
        "https://youtube.com/feed/trending",
    ])
    def test_invalid_urls(self, url: str) -> None:
        assert validate_url(url) is False


class TestFormatTimestamp:
    def test_zero(self) -> None:
        assert format_timestamp(0.0) == "00:00:00,000"

    def test_seconds_only(self) -> None:
        assert format_timestamp(5.5) == "00:00:05,500"

    def test_minutes(self) -> None:
        assert format_timestamp(125.75) == "00:02:05,750"

    def test_hours(self) -> None:
        assert format_timestamp(3661.001) == "01:01:01,001"

    def test_full_duration(self) -> None:
        assert format_timestamp(45296.999) == "12:34:56,999"


class TestGetOutputBasename:
    def test_normal_title(self) -> None:
        result = get_output_basename("Hello World", "BV123")
        assert result == "Hello World"

    def test_strips_invalid_chars(self) -> None:
        result = get_output_basename('File: "Name" <test>?', "BV123")
        assert ":" not in result
        assert '"' not in result
        assert "<" not in result
        assert ">" not in result
        assert "?" not in result

    def test_collapses_whitespace(self) -> None:
        result = get_output_basename("Hello    World", "BV123")
        assert result == "Hello World"

    def test_falls_back_to_video_id(self) -> None:
        result = get_output_basename('<>:"/\\|?*', "BV123456")
        assert result == "BV123456"

    def test_truncates_long_name(self) -> None:
        result = get_output_basename("A" * 200, "BV123")
        assert len(result) <= 100

    def test_empty_title_no_id(self) -> None:
        result = get_output_basename("", "")
        assert result == "untitled"


class TestCheckDependencies:
    def test_returns_list(self) -> None:
        result = check_dependencies()
        assert isinstance(result, list)

    def test_all_strings(self) -> None:
        result = check_dependencies()
        for item in result:
            assert isinstance(item, str)
