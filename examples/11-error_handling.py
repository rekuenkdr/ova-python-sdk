"""Proper error handling patterns for every exception type.

Demonstrates how to catch and handle each OVA SDK exception.
No optional dependencies required.
"""

import sys

from ova_sdk import (
    OVA,
    OVAAuthenticationError,
    OVAConnectionError,
    OVAError,
    OVARequestError,
    OVAServerNotReady,
    OVATimeoutError,
)

LANGUAGE = "en"

# -- 1. Authentication error: bad API key --
print("=== OVAAuthenticationError ===")
try:
    # If the server has OVA_API_KEY set, a wrong key triggers 401 immediately
    bad_key_client = OVA(api_key="wrong-key")
    bad_key_client.ready()
except OVAAuthenticationError as e:
    print(f"  Caught OVAAuthenticationError: {e}")
except (OVAConnectionError, OVAServerNotReady):
    print("  Skipped — server not available (start OVA with OVA_API_KEY to test)")

# -- 2. Connection error: server not running --
print("=== OVAConnectionError ===")
try:
    bad_client = OVA(base_url="http://localhost:9999")
    bad_client.wait_until_ready(timeout=3, poll_interval=1)
except OVAServerNotReady:
    print("  Caught OVAServerNotReady: server at :9999 never became ready")
except OVAConnectionError as e:
    print(f"  Caught OVAConnectionError: {e}")

# -- 3. Server not ready: timeout waiting for warmup --
print("\n=== OVAServerNotReady (wait_until_ready timeout) ===")
try:
    # Very short timeout to demonstrate the exception
    client = OVA(base_url="http://localhost:9999")
    client.wait_until_ready(timeout=1, poll_interval=0.5)
except OVAServerNotReady as e:
    print(f"  Caught OVAServerNotReady: {e}")
except OVAConnectionError:
    print("  Caught OVAConnectionError (server not reachable at all)")

# -- 4. Request error: invalid parameters (requires running server) --
print("\n=== OVARequestError (invalid voice) ===")
try:
    client = OVA()
    client.wait_until_ready(timeout=5)

    # Invalid voice name should trigger a 400 error
    audio = client.tts.generate("Hello", voice="nonexistent_voice_xyz", language=LANGUAGE)
    audio.to_bytes()  # force consumption to trigger the error
except OVARequestError as e:
    print(f"  Caught OVARequestError: HTTP {e.status_code} — {e}")
except (OVAConnectionError, OVAServerNotReady):
    print("  Skipped — server not available (start OVA to test this case)")

# -- 5. Timeout error --
# Note: OVATimeoutError is raised when the *initial* request times out.
# During streaming iteration, httpx.ReadTimeout may surface directly.
# Catching both covers all timeout scenarios.
print("\n=== OVATimeoutError ===")
try:
    import httpx

    client = OVA(timeout=0.001)
    client.wait_until_ready(timeout=5)
    voices = client.voices(language=LANGUAGE)
    voice = voices[0] if voices else "default"
    audio = client.tts.generate(
        "This sentence is long enough to take some time to generate.",
        voice=voice,
        language=LANGUAGE,
    )
    audio.to_bytes()
except OVATimeoutError as e:
    print(f"  Caught OVATimeoutError: {e}")
except httpx.TimeoutException as e:
    # Timeout during stream iteration (after connection was established)
    print(f"  Caught httpx.TimeoutException during streaming: {type(e).__name__}")
except (OVAConnectionError, OVAServerNotReady):
    print("  Skipped — server not available (start OVA to test this case)")
except OVARequestError:
    print("  Got a request error instead (server responded before timeout)")

# -- 6. Catch-all with base exception --
print("\n=== OVAError (catch-all) ===")
try:
    client = OVA(base_url="http://localhost:9999")
    client.wait_until_ready(timeout=1, poll_interval=0.5)
except OVAError as e:
    # Catches any OVA exception — useful as a fallback
    print(f"  Caught {type(e).__name__}: {e}")

print("\nDone!")
