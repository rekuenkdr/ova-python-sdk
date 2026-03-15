"""Unit tests for _base.py — URL resolution, headers, raise_for_status."""

import os

import httpx
import pytest

from ova_sdk._base import (
    _build_headers,
    _resolve_api_key,
    _resolve_base_url,
    raise_for_status,
    ws_url,
)
from ova_sdk._errors import OVAAuthenticationError, OVARequestError


class TestResolveBaseUrl:
    def test_default(self, monkeypatch):
        monkeypatch.delenv("OVA_BASE_URL", raising=False)
        assert _resolve_base_url(None) == "http://localhost:5173"

    def test_explicit(self, monkeypatch):
        monkeypatch.delenv("OVA_BASE_URL", raising=False)
        assert _resolve_base_url("http://myhost:8080") == "http://myhost:8080"

    def test_env(self, monkeypatch):
        monkeypatch.setenv("OVA_BASE_URL", "http://envhost:9000/")
        assert _resolve_base_url(None) == "http://envhost:9000"

    def test_explicit_beats_env(self, monkeypatch):
        monkeypatch.setenv("OVA_BASE_URL", "http://envhost:9000")
        assert _resolve_base_url("http://explicit:1234") == "http://explicit:1234"

    def test_strips_trailing_slash(self):
        assert _resolve_base_url("http://host:5000/") == "http://host:5000"


class TestResolveApiKey:
    def test_none_default(self, monkeypatch):
        monkeypatch.delenv("OVA_API_KEY", raising=False)
        assert _resolve_api_key(None) is None

    def test_explicit(self, monkeypatch):
        monkeypatch.delenv("OVA_API_KEY", raising=False)
        assert _resolve_api_key("sk-test") == "sk-test"

    def test_env(self, monkeypatch):
        monkeypatch.setenv("OVA_API_KEY", "sk-from-env")
        assert _resolve_api_key(None) == "sk-from-env"

    def test_explicit_beats_env(self, monkeypatch):
        monkeypatch.setenv("OVA_API_KEY", "sk-from-env")
        assert _resolve_api_key("sk-explicit") == "sk-explicit"

    def test_empty_env_is_none(self, monkeypatch):
        monkeypatch.setenv("OVA_API_KEY", "")
        assert _resolve_api_key(None) is None


class TestBuildHeaders:
    def test_no_key(self):
        assert _build_headers(None) == {}

    def test_with_key(self):
        h = _build_headers("sk-test")
        assert h == {"Authorization": "Bearer sk-test"}


class TestWsUrl:
    def test_http_to_ws(self):
        assert ws_url("http://localhost:5173", "/v1/asr", None) == "ws://localhost:5173/v1/asr"

    def test_https_to_wss(self):
        assert ws_url("https://example.com", "/v1/asr", None) == "wss://example.com/v1/asr"

    def test_with_api_key(self):
        result = ws_url("http://localhost:5173", "/v1/asr", "sk-key")
        assert result == "ws://localhost:5173/v1/asr?api_key=sk-key"

    def test_api_key_special_chars(self):
        result = ws_url("http://localhost:5173", "/v1/asr", "key with spaces")
        assert "api_key=key%20with%20spaces" in result


class TestRaiseForStatus:
    def test_200_ok(self):
        resp = httpx.Response(200)
        raise_for_status(resp)  # should not raise

    def test_401_auth_error(self):
        resp = httpx.Response(401)
        with pytest.raises(OVAAuthenticationError):
            raise_for_status(resp)

    def test_500_request_error(self):
        resp = httpx.Response(500, text="Internal Server Error")
        with pytest.raises(OVARequestError) as exc_info:
            raise_for_status(resp)
        assert exc_info.value.status_code == 500

    def test_404_request_error(self):
        resp = httpx.Response(404, text="Not Found")
        with pytest.raises(OVARequestError) as exc_info:
            raise_for_status(resp)
        assert exc_info.value.status_code == 404
