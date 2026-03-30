"""
tests/config/elastic_setup_test.py

Unit tests for config/elastic_setup.py — get_config():
  - Happy path reads all key=value pairs correctly
  - Comments (lines starting with #) are skipped
  - Blank lines are skipped
  - Values with embedded '=' are preserved (maxsplit=1)
  - FileNotFoundError is swallowed and an empty dict is returned
  - Keys and values are stripped of surrounding whitespace

Note: elastic_setup.py constructs the path as:
    BASE_PATH + "/resources/elastic-config.properties"
so every tmp_path fixture must write the file into a resources/ subdirectory
and BASE_PATH must be patched to str(tmp_path).
"""
from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest


# ---------------------------------------------------------------------------
# Helper — write props file into tmp_path/resources/elastic-config.properties
# ---------------------------------------------------------------------------

def _write_props(tmp_path: Path, content: str) -> Path:
    """Create resources/ subdir and write properties content."""
    resources = tmp_path / "resources"
    resources.mkdir(parents=True, exist_ok=True)
    props = resources / "elastic-config.properties"
    props.write_text(content)
    return props


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------

class TestGetConfigHappyPath:
    def test_reads_all_key_value_pairs(self, tmp_path):
        _write_props(tmp_path,
            "request_timeout=60\n"
            "retry_on_timeout=True\n"
            "max_retries=3\n"
            "http_compress=True\n"
            "connections_per_node=10\n"
        )
        with patch("config.elastic_setup.BASE_PATH", str(tmp_path)):
            from config.elastic_setup import get_config
            config = get_config()

        assert config["request_timeout"] == "60"
        assert config["retry_on_timeout"] == "True"
        assert config["max_retries"] == "3"
        assert config["http_compress"] == "True"
        assert config["connections_per_node"] == "10"

    def test_returns_all_five_standard_keys(self, tmp_path):
        _write_props(tmp_path,
            "request_timeout=60\n"
            "retry_on_timeout=True\n"
            "max_retries=3\n"
            "http_compress=True\n"
            "connections_per_node=10\n"
        )
        with patch("config.elastic_setup.BASE_PATH", str(tmp_path)):
            from config.elastic_setup import get_config
            config = get_config()

        expected_keys = {
            "request_timeout",
            "retry_on_timeout",
            "max_retries",
            "http_compress",
            "connections_per_node",
        }
        assert expected_keys.issubset(set(config.keys()))


# ---------------------------------------------------------------------------
# Comments and blank lines
# ---------------------------------------------------------------------------

class TestGetConfigFiltering:
    def test_comment_lines_skipped(self, tmp_path):
        _write_props(tmp_path,
            "# This is a comment\n"
            "request_timeout=30\n"
        )
        with patch("config.elastic_setup.BASE_PATH", str(tmp_path)):
            from config.elastic_setup import get_config
            config = get_config()

        assert "# This is a comment" not in config
        assert config["request_timeout"] == "30"

    def test_blank_lines_skipped(self, tmp_path):
        _write_props(tmp_path,
            "\n"
            "request_timeout=45\n"
            "\n"
            "max_retries=5\n"
        )
        with patch("config.elastic_setup.BASE_PATH", str(tmp_path)):
            from config.elastic_setup import get_config
            config = get_config()

        assert "" not in config
        assert config["request_timeout"] == "45"
        assert config["max_retries"] == "5"

    def test_inline_comment_preserved_in_value(self, tmp_path):
        """
        Only leading-'#' lines are treated as comments.
        A '#' embedded in a value must be left intact.
        """
        _write_props(tmp_path, "my_key=value#with_hash\n")
        with patch("config.elastic_setup.BASE_PATH", str(tmp_path)):
            from config.elastic_setup import get_config
            config = get_config()

        assert config["my_key"] == "value#with_hash"


# ---------------------------------------------------------------------------
# Values containing '='
# ---------------------------------------------------------------------------

class TestGetConfigEqualSignInValue:
    def test_value_with_equals_sign_preserved(self, tmp_path):
        """Values that contain '=' (e.g. Base64 API keys) must not be split."""
        _write_props(tmp_path, "api_key=abc=def==ghi\n")
        with patch("config.elastic_setup.BASE_PATH", str(tmp_path)):
            from config.elastic_setup import get_config
            config = get_config()

        assert config["api_key"] == "abc=def==ghi"


# ---------------------------------------------------------------------------
# Whitespace stripping
# ---------------------------------------------------------------------------

class TestGetConfigWhitespace:
    def test_keys_stripped(self, tmp_path):
        _write_props(tmp_path, "  request_timeout  =  60  \n")
        with patch("config.elastic_setup.BASE_PATH", str(tmp_path)):
            from config.elastic_setup import get_config
            config = get_config()

        assert "request_timeout" in config
        assert config["request_timeout"] == "60"

    def test_values_stripped(self, tmp_path):
        _write_props(tmp_path, "max_retries=  3  \n")
        with patch("config.elastic_setup.BASE_PATH", str(tmp_path)):
            from config.elastic_setup import get_config
            config = get_config()

        assert config["max_retries"] == "3"


# ---------------------------------------------------------------------------
# Missing file — FileNotFoundError must be swallowed
# ---------------------------------------------------------------------------

class TestGetConfigMissingFile:
    def test_returns_empty_dict_on_file_not_found(self, tmp_path):
        # Point BASE_PATH at a directory that has no resources/ subdir
        non_existent = tmp_path / "no_such_dir"
        with patch("config.elastic_setup.BASE_PATH", str(non_existent)):
            from config.elastic_setup import get_config
            config = get_config()

        assert config == {}

    def test_does_not_raise_on_file_not_found(self, tmp_path):
        non_existent = tmp_path / "missing"
        with patch("config.elastic_setup.BASE_PATH", str(non_existent)):
            from config.elastic_setup import get_config
            try:
                get_config()
            except FileNotFoundError:
                pytest.fail("get_config() should not propagate FileNotFoundError")


# ---------------------------------------------------------------------------
# Empty file
# ---------------------------------------------------------------------------

class TestGetConfigEmptyFile:
    def test_empty_file_returns_empty_dict(self, tmp_path):
        _write_props(tmp_path, "")
        with patch("config.elastic_setup.BASE_PATH", str(tmp_path)):
            from config.elastic_setup import get_config
            config = get_config()

        assert config == {}

    def test_only_comments_returns_empty_dict(self, tmp_path):
        _write_props(tmp_path, "# comment one\n# comment two\n")
        with patch("config.elastic_setup.BASE_PATH", str(tmp_path)):
            from config.elastic_setup import get_config
            config = get_config()

        assert config == {}