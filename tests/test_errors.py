"""Unit tests for _errors.py — exception hierarchy."""

from ova_sdk._errors import (
    OVAAuthenticationError,
    OVAConnectionError,
    OVAError,
    OVARequestError,
    OVAServerNotReady,
    OVATimeoutError,
)


class TestExceptionHierarchy:
    def test_all_inherit_from_ova_error(self):
        assert issubclass(OVAAuthenticationError, OVAError)
        assert issubclass(OVAConnectionError, OVAError)
        assert issubclass(OVAServerNotReady, OVAError)
        assert issubclass(OVARequestError, OVAError)
        assert issubclass(OVATimeoutError, OVAError)

    def test_ova_error_is_exception(self):
        assert issubclass(OVAError, Exception)

    def test_request_error_has_status_code(self):
        e = OVARequestError(404, "Not Found")
        assert e.status_code == 404
        assert "404" in str(e)
        assert "Not Found" in str(e)

    def test_request_error_empty_message(self):
        e = OVARequestError(500)
        assert e.status_code == 500
        assert "500" in str(e)

    def test_catch_specific_with_base(self):
        """OVAError catch-all should catch all subtypes."""
        for exc_cls in [OVAAuthenticationError, OVAConnectionError, OVAServerNotReady, OVATimeoutError]:
            try:
                raise exc_cls("test")
            except OVAError:
                pass  # expected

    def test_request_error_caught_by_base(self):
        try:
            raise OVARequestError(400, "Bad Request")
        except OVAError:
            pass  # expected

    def test_isinstance_checks(self):
        e = OVAAuthenticationError("bad key")
        assert isinstance(e, OVAError)
        assert isinstance(e, OVAAuthenticationError)
        assert not isinstance(e, OVARequestError)
