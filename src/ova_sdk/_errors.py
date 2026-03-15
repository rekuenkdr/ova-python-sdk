"""Exception hierarchy for OVA SDK."""


class OVAError(Exception):
    """Base exception for all OVA SDK errors."""


class OVAAuthenticationError(OVAError):
    """API key is missing or invalid (401)."""


class OVAConnectionError(OVAError):
    """Cannot reach the OVA server."""


class OVAServerNotReady(OVAError):
    """Server is still warming up (503)."""


class OVARequestError(OVAError):
    """Server returned an error response (4xx/5xx)."""

    def __init__(self, status_code: int, message: str = ""):
        self.status_code = status_code
        super().__init__(f"HTTP {status_code}: {message}" if message else f"HTTP {status_code}")


class OVATimeoutError(OVAError):
    """Request timed out."""
