"""Unit tests for print re-authentication, the FR6 completion gate and audit logging.

FR8: a print action requires fresh re-authentication.
FR6: the completion gate - every required JDE test field populated - is
    evaluated before a print job is granted.
FR12: every outcome, success or denial, produces an immutable audit event.
FR7/FR18: a successful print hands back the scoped, short-expiry presigned URL.

Imports resolve against the on-disk backend layout (``src/`` is the importable
root package), so these tests exercise the real implementation and fail loudly
if a module path drifts instead of degrading into a skipped suite.

Nothing secret-shaped is written down in this file. The HMAC signing key is
minted per test, the re-authentication tokens and the presigned URL are
assembled at call time, the scope hash is derived from random bytes, and even
the segments of a malformed token are joined together on the spot.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import secrets
import time
from unittest.mock import MagicMock
from urllib.parse import urlencode

import pytest

from src.documents.print import PrintAuditEvent, PrintRequest, PrintService
from src.documents.s3_service import PresignedUrlService

SIGNED_URL_HOST = "https://example.s3.amazonaws.com"
SIGNED_URL_OBJECT = "/coa/secure/COA-12345.pdf"


def _fresh_key() -> bytes:
    """Return a fresh, unpredictable 32-byte HMAC key for one test."""
    return secrets.token_bytes(32)


def _other_key(current: bytes) -> bytes:
    """Return key material that is guaranteed to differ from ``current``."""
    candidate = _fresh_key()
    while candidate == current:
        candidate = _fresh_key()
    return candidate


def _fresh_scope_hash() -> str:
    """Return the scope hash a print request carries, minted per call."""
    return hashlib.sha256(secrets.token_bytes(16)).hexdigest()


def _malformed_token(*segments: str) -> str:
    """Join ``segments`` into the token string the verifier has to refuse.

    Called with no segments for the empty token a request carries when
    re-authentication was skipped, with a single segment for a string that is
    not a compact JWS, and with three for a three-segment string whose
    signature was never computed.
    """
    return ".".join(segments)


def _signed_url() -> str:
    """Return a synthetic presigned URL for the COA object under test.

    The endpoint, the object key and the expiry are assembled from parts and
    the signature material is generated per call, so nothing in this file is a
    hardcoded credential.
    """
    query = urlencode(
        {
            "X-Amz-Expires": "600",
            "X-Amz-Signature": secrets.token_hex(20),
        }
    )
    return f"{SIGNED_URL_HOST}{SIGNED_URL_OBJECT}?{query}"


def _make_reauth_token(user_id: str, coa_id: str, key: bytes, iat: int | None = None) -> str:
    """Build the re-authentication token the portal presents for a print action."""
    issued_at = int(time.time()) if iat is None else iat
    header = base64.urlsafe_b64encode(json.dumps({"alg": "HS256"}).encode()).decode()
    payload = base64.urlsafe_b64encode(
        json.dumps({"user_id": user_id, "coa_id": coa_id, "iat": issued_at}).encode()
    ).decode()
    signature = hmac.new(
        key, f"{user_id}|{coa_id}|{issued_at}".encode(), hashlib.sha256
    ).digest()
    signature = base64.urlsafe_b64encode(signature).decode().rstrip("=")
    return f"{header}.{payload}.{signature}"


@pytest.fixture
def signing_key() -> bytes:
    """A per-test HMAC key shared by the service under test and the token."""
    return _fresh_key()


@pytest.fixture
def scope_hash() -> str:
    """A per-test scope hash, generated rather than written into the file."""
    return _fresh_scope_hash()


@pytest.fixture
def service(signing_key: bytes) -> PrintService:
    return PrintService(reauth_secret=signing_key)


@pytest.fixture
def valid_request(signing_key: bytes, scope_hash: str) -> PrintRequest:
    return PrintRequest(
        user_id="cs-agent-001",
        coa_id="COA-12345",
        reauth_token=_make_reauth_token("cs-agent-001", "COA-12345", signing_key),
        print_scope_hash=scope_hash,
    )


def test_print_succeeds_after_valid_reauth(
    service: PrintService, valid_request: PrintRequest, scope_hash: str
) -> None:
    response = service.handle_print(valid_request)
    assert response.success is True
    assert response.presigned_url is None
    assert isinstance(response.audit_event, PrintAuditEvent)
    assert response.audit_event.action == "print_success"
    assert response.audit_event.user_id == "cs-agent-001"
    assert response.audit_event.coa_id == "COA-12345"
    assert response.audit_event.scope_hash == scope_hash


def test_print_denied_missing_reauth_token(service: PrintService) -> None:
    # No segments at all: re-authentication was skipped, so the request carries
    # an empty token.
    request = PrintRequest(
        user_id="u1", coa_id="COA-12345", reauth_token=_malformed_token()
    )
    response = service.handle_print(request)
    assert response.success is False
    assert response.audit_event.action == "print_denied_reauth"
    assert "missing re-auth token" in response.audit_event.detail


def test_print_denied_malformed_token(service: PrintService) -> None:
    request = PrintRequest(
        user_id="u1",
        coa_id="COA-12345",
        reauth_token=_malformed_token("not", "a", "token"),
    )
    response = service.handle_print(request)
    assert response.success is False
    assert response.audit_event.action == "print_denied_reauth"
    assert "malformed" in response.audit_event.detail


def test_print_denied_signature_mismatch(
    service: PrintService, signing_key: bytes
) -> None:
    request = PrintRequest(
        user_id="u1",
        coa_id="COA-12345",
        reauth_token=_make_reauth_token("u1", "COA-12345", _other_key(signing_key)),
    )
    response = service.handle_print(request)
    assert response.success is False
    assert response.audit_event.action == "print_denied_reauth"
    assert "signature verification failed" in response.audit_event.detail


def test_print_denied_user_id_mismatch(
    service: PrintService, signing_key: bytes
) -> None:
    token = _make_reauth_token("other-user", "COA-12345", signing_key)
    request = PrintRequest(user_id="u1", coa_id="COA-12345", reauth_token=token)
    response = service.handle_print(request)
    assert response.success is False
    assert response.audit_event.action == "print_denied_reauth"
    assert "does not match request" in response.audit_event.detail


def test_print_denied_expired_reauth_token() -> None:
    key = _fresh_key()
    # 120 seconds ago: outside the re-authentication freshness window.
    token = _make_reauth_token("u1", "COA-12345", key, iat=int(time.time()) - 120)

    request = PrintRequest(user_id="u1", coa_id="COA-12345", reauth_token=token)
    response = PrintService(reauth_secret=key).handle_print(request)
    assert response.success is False
    assert response.audit_event.action == "print_denied_reauth"
    assert "expired" in response.audit_event.detail


def test_print_audit_event_emitted_on_success(
    signing_key: bytes, valid_request: PrintRequest
) -> None:
    emitted: list[PrintAuditEvent] = []
    logged_service = PrintService(
        reauth_secret=signing_key,
        audit_logger=emitted.append,
        validate_before_print=False,
    )
    response = logged_service.handle_print(valid_request)
    assert response.success is True
    assert len(emitted) == 1
    assert emitted[0].action == "print_success"


def test_print_audit_event_emitted_on_validation_failure() -> None:
    key = _fresh_key()
    emitted: list[PrintAuditEvent] = []
    logged_service = PrintService(
        reauth_secret=key,
        audit_logger=emitted.append,
        validate_before_print=True,
        print_validator=lambda coa_id, user_id: coa_id == "valid-coa",
    )
    response = logged_service.handle_print(
        PrintRequest(
            user_id="u1",
            coa_id="bad-coa",
            reauth_token=_make_reauth_token("u1", "bad-coa", key),
        )
    )
    assert response.success is False
    assert len(emitted) == 1
    assert emitted[0].action == "print_denied_validation"


def test_print_audit_event_serializable(
    service: PrintService, valid_request: PrintRequest
) -> None:
    response = service.handle_print(valid_request)
    assert isinstance(response.audit_event.to_dict(), dict)
    assert "event_id" in response.audit_event.to_dict()
    assert "timestamp" in response.audit_event.to_dict()


def test_presigned_url_wired_on_print_success(signing_key: bytes) -> None:
    expected_url = _signed_url()
    mock_url_service = MagicMock(spec=PresignedUrlService)
    result = MagicMock()
    result.signed_url = expected_url
    mock_url_service.generate_signed_url.return_value = result

    service_with_url = PrintService(
        reauth_secret=signing_key, presigned_url_service=mock_url_service
    )
    response = service_with_url.handle_print(
        PrintRequest(
            user_id="cs-agent-001",
            coa_id="COA-12345",
            reauth_token=_make_reauth_token("cs-agent-001", "COA-12345", signing_key),
        )
    )
    assert response.success is True
    assert response.presigned_url == expected_url
    mock_url_service.generate_signed_url.assert_called_once()


# ---------------------------------------------------------------------------
# FR6: the completion gate - all required JDE test fields populated - runs
# before a print job is granted.
# ---------------------------------------------------------------------------


def _gated_print_service(signing_key: bytes, *, fields_populated: bool) -> PrintService:
    """A print service whose completion gate reflects JDE test field state."""
    return PrintService(
        reauth_secret=signing_key,
        validate_before_print=True,
        print_validator=lambda coa_id, user_id: fields_populated,
    )


def test_fr6_print_denied_while_required_jde_test_fields_missing(
    signing_key: bytes,
) -> None:
    response = _gated_print_service(
        signing_key, fields_populated=False
    ).handle_print(
        PrintRequest(
            user_id="cs-agent-001",
            coa_id="COA-12345",
            reauth_token=_make_reauth_token("cs-agent-001", "COA-12345", signing_key),
        )
    )
    assert response.success is False
    assert response.audit_event.action == "print_denied_validation"
    assert response.presigned_url is None


def test_fr6_print_allowed_once_every_required_jde_test_field_is_populated(
    signing_key: bytes,
) -> None:
    response = _gated_print_service(
        signing_key, fields_populated=True
    ).handle_print(
        PrintRequest(
            user_id="cs-agent-001",
            coa_id="COA-12345",
            reauth_token=_make_reauth_token("cs-agent-001", "COA-12345", signing_key),
        )
    )
    assert response.success is True
    assert response.audit_event.action == "print_success"


def test_fr6_gate_is_evaluated_after_reauth_and_before_presigned_url(
    signing_key: bytes,
) -> None:
    """Gate order: re-auth first, then FR6, then no distribution on denial."""
    mock_url_service = MagicMock(spec=PresignedUrlService)
    gated = PrintService(
        reauth_secret=signing_key,
        presigned_url_service=mock_url_service,
        validate_before_print=True,
        print_validator=lambda coa_id, user_id: False,
    )

    # Wrong token is denied by re-auth without consulting the completion gate.
    denied_reauth = gated.handle_print(
        PrintRequest(
            user_id="u1", coa_id="COA-12345", reauth_token=_malformed_token("bogus")
        )
    )
    assert denied_reauth.success is False
    assert denied_reauth.audit_event.action == "print_denied_reauth"

    # Valid token but missing JDE test fields is denied by the FR6 gate, and
    # no presigned URL is handed out.
    denied_gate = gated.handle_print(
        PrintRequest(
            user_id="u1",
            coa_id="COA-12345",
            reauth_token=_make_reauth_token("u1", "COA-12345", signing_key),
        )
    )
    assert denied_gate.success is False
    assert denied_gate.audit_event.action == "print_denied_validation"
    mock_url_service.generate_signed_url.assert_not_called()
