"""Unit tests for _chat.py — image resolution."""

import base64
import struct
import tempfile
from pathlib import Path

import pytest

from ova_sdk._chat import _resolve_image


class TestResolveImage:
    def test_data_url_passthrough(self):
        url = "data:image/png;base64,iVBOR..."
        assert _resolve_image(url) == url

    def test_real_png_file(self, tmp_path):
        """Create a minimal 1x1 PNG and resolve it."""
        # Minimal valid 1x1 white PNG
        png_data = (
            b"\x89PNG\r\n\x1a\n"  # signature
            b"\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01\x08\x02\x00\x00\x00\x90wS\xde"
            b"\x00\x00\x00\x0cIDATx\x9cc\xf8\x0f\x00\x00\x01\x01\x00\x05\x18\xd8N"
            b"\x00\x00\x00\x00IEND\xaeB`\x82"
        )
        png_file = tmp_path / "test.png"
        png_file.write_bytes(png_data)

        result = _resolve_image(png_file)
        assert result.startswith("data:image/png;base64,")

        # Verify round-trip
        b64_part = result.split(",", 1)[1]
        decoded = base64.b64decode(b64_part)
        assert decoded == png_data

    def test_real_jpg_file(self, tmp_path):
        jpg_file = tmp_path / "test.jpg"
        jpg_file.write_bytes(b"\xff\xd8\xff\xe0" + b"\x00" * 10)
        result = _resolve_image(jpg_file)
        assert result.startswith("data:image/jpeg;base64,")

    def test_string_path(self, tmp_path):
        png_file = tmp_path / "test.png"
        png_file.write_bytes(b"\x89PNG" + b"\x00" * 10)
        result = _resolve_image(str(png_file))
        assert result.startswith("data:image/png;base64,")

    def test_missing_file_raises(self):
        with pytest.raises(ValueError, match="not found"):
            _resolve_image(Path("/nonexistent/image.png"))

    def test_missing_string_path_raises(self):
        with pytest.raises(ValueError, match="not found"):
            _resolve_image("/nonexistent/image.png")

    def test_raw_base64_passthrough(self):
        """A string that doesn't look like a path or data URL is treated as raw base64."""
        b64 = base64.b64encode(b"fake image data").decode()
        assert _resolve_image(b64) == b64
