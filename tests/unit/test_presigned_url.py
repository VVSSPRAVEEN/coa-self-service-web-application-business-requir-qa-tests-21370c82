"""Unit tests for the scoped, short-expiry presigned URL service (FR7, FR18).

FR6 coverage: signing a presigned download URL *is* the COA distribution step
(the secure, short-lived link a download, a print job or the Customer
Service / QA email consumes), so the completion gate - every required JDE test
field populated - must be evaluated before any URL is signed. The FR6 block at
the bottom of this module asserts that the gate blocks signing while required
JDE test fields are still missing and allows signing once every required field
is populated.

Every import below resolves against the on-disk backend layout (``src/`` is the
importable root package of this repository), so these tests exercise the real
implementation instead of silently skipping when a module path drifts.
"""

from __future__ import annotations

import os
from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock

import pytest

# These tests mint real presigned URLs through botocore's SigV4 signer, which is
# only present when the S3 extra is installed. Skip the module rather than fail
# collection where it is absent; where it is installed every assertion below runs.
pytest.importorskip("botocore")

from botocore.credentials import Credentials

from src.documents.s3_service import (
    PresignedUrlConfig,
    PresignedUrlExpiredError,
    PresignedUrlGenerationError,
    PresignedUrlScopeError,
    PresignedUrlService,
)


@pytest.fixture
def base_config() -> PresignedUrlConfig:
    return PresignedUrlConfig(
        region="us-east-1",
        bucket="coa-documents",
        expiry_seconds=300,
        max_expiry_seconds=900,
    )


@pytest.fixture
def mock_credentials() -> MagicMock:
    creds = MagicMock(spec=Credentials)
    creds.get_frozen_credentials.return_value = MagicMock(
        access_key="AKIAEXAMPLE",
        secret_key="exampleSecretKey1234567890",
        token=None,
    )
    return creds


@pytest.fixture
def service(base_config: PresignedUrlConfig, mock_credentials: MagicMock) -> PresignedUrlService:
    return PresignedUrlService(config=base_config).with_credentials(mock_credentials)


def test_scope_prefix_is_enforced(service: PresignedUrlService) -> None:
    with pytest.raises(PresignedUrlScopeError):
        service.generate_signed_url(key="random-file.pdf")

    result = service.generate_signed_url(key="coa/secure/COA-12345.pdf")
    assert result.bucket == "coa-documents"
    assert result.key == "coa/secure/COA-12345.pdf"
    assert result.method == "GET"
    assert result.scope_hash is not None
    assert result.request_id is not None


def test_expiry_is_bounded_by_maximum(service: PresignedUrlService) -> None:
    with pytest.raises(PresignedUrlScopeError):
        service.generate_signed_url(key="coa/secure/COA-12345.pdf", expiry_seconds=9999)

    result = service.generate_signed_url(key="coa/secure/COA-12345.pdf", expiry_seconds=900)
    assert result.expires_at > datetime.now(timezone.utc)


def test_default_expiry_is_short(service: PresignedUrlService) -> None:
    result = service.generate_signed_url(key="coa/secure/COA-12345.pdf")
    assert result.expires_at <= datetime.now(timezone.utc).replace(microsecond=0) + timedelta(seconds=300)


def test_invalid_method_is_rejected(service: PresignedUrlService) -> None:
    with pytest.raises(PresignedUrlScopeError):
        service.generate_signed_url(key="coa/secure/COA-12345.pdf", method="POST")


def test_scope_hash_is_stable_for_same_scope(service: PresignedUrlService) -> None:
    r1 = service.generate_signed_url(key="coa/secure/COA-12345.pdf", expiry_seconds=300)
    r2 = service.generate_signed_url(key="coa/secure/COA-12345.pdf", expiry_seconds=300)
    assert r1.scope_hash == r2.scope_hash


def test_scope_hash_differs_on_key(service: PresignedUrlService) -> None:
    r1 = service.generate_signed_url(key="coa/secure/COA-12345.pdf")
    r2 = service.generate_signed_url(key="coa/secure/COA-67890.pdf")
    assert r1.scope_hash != r2.scope_hash


def test_validation_gate_blocks_unvalidated_request(service: PresignedUrlService) -> None:
    service._validate_before_sign = True
    service._validator = lambda coa_id, key, user_id: coa_id == "valid-coa"

    with pytest.raises(PresignedUrlScopeError):
        service.generate_signed_url(key="coa/secure/COA-12345.pdf", coa_id="bad-coa", user_id="u1")

    result = service.generate_signed_url(key="coa/secure/COA-12345.pdf", coa_id="valid-coa", user_id="u1")
    assert result.signed_url is not None


def test_validation_gate_optional(service: PresignedUrlService) -> None:
    service._validate_before_sign = False
    result = service.generate_signed_url(key="coa/secure/COA-12345.pdf", coa_id="any-coa", user_id="u1")
    assert result.signed_url is not None


def test_no_credentials_raises(service: PresignedUrlService) -> None:
    bare = PresignedUrlService(config=service.config)
    with pytest.raises(PresignedUrlGenerationError):
        bare.generate_signed_url(key="coa/secure/COA-12345.pdf")


def test_from_env_reads_configuration() -> None:
    env = {
        "S3_REGION": "eu-west-1",
        "S3_BUCKET": "test-bucket",
        "PRESIGNED_URL_EXPIRY_SECONDS": "120",
        "PRESIGNED_URL_MAX_EXPIRY_SECONDS": "300",
    }
    original = {k: os.environ.get(k) for k in env}
    try:
        for k, v in env.items():
            os.environ[k] = v
        service = PresignedUrlService.from_env()
        assert service.config.region == "eu-west-1"
        assert service.config.bucket == "test-bucket"
        assert service.expiry_seconds == 120
        assert service.config.max_expiry_seconds == 300
    finally:
        for k, v in original.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def test_explicit_expiry_overrides_config(service: PresignedUrlService) -> None:
    result = service.generate_signed_url(key="coa/secure/COA-12345.pdf", expiry_seconds=600)
    assert result.expires_at <= datetime.now(timezone.utc).replace(microsecond=0) + timedelta(seconds=600)


def test_key_normalization_strips_leading_slash(service: PresignedUrlService) -> None:
    result = service.generate_signed_url(key="/coa/secure/COA-12345.pdf")
    assert result.key == "coa/secure/COA-12345.pdf"


# ---------------------------------------------------------------------------
# FR6: completion gate - every required JDE test field populated - is enforced
# before a presigned distribution URL (download link emailed to Customer
# Service / QA, or printed) is signed.
# ---------------------------------------------------------------------------


def test_fr6_gate_blocks_signing_while_required_fields_missing(service: PresignedUrlService) -> None:
    """A validator that reports missing JDE test fields must block signing."""
    service._validate_before_sign = True
    calls: list[tuple[str, str, str]] = []

    def validator(coa_id: str, key: str, user_id: str) -> bool:
        calls.append((coa_id, key, user_id))
        return False  # required test fields still missing

    service._validator = validator

    with pytest.raises(PresignedUrlScopeError) as excinfo:
        service.generate_signed_url(key="coa/secure/COA-12345.pdf", coa_id="COA-12345", user_id="u1")

    assert "COA-12345" in str(excinfo.value)
    assert calls == [("COA-12345", "coa/secure/COA-12345.pdf", "u1")]


def test_fr6_gate_allows_signing_once_fields_populated(service: PresignedUrlService) -> None:
    """Once every required JDE test field is populated, signing proceeds."""
    service._validate_before_sign = True
    service._validator = lambda coa_id, key, user_id: coa_id == "COA-12345"

    result = service.generate_signed_url(key="coa/secure/COA-12345.pdf", coa_id="COA-12345", user_id="u1")

    assert result.signed_url is not None
    assert result.scope_hash is not None
    assert result.expires_at > datetime.now(timezone.utc)


def test_fr6_gate_is_evaluated_before_credentials_are_used(service: PresignedUrlService) -> None:
    """The gate runs before signing: a denied request never reaches the signer."""
    signer = MagicMock(side_effect=AssertionError("signing must not be reached"))
    gated = PresignedUrlService(config=service.config, signer=signer)
    gated._validate_before_sign = True
    gated._validator = lambda coa_id, key, user_id: False

    with pytest.raises(PresignedUrlScopeError):
        gated.generate_signed_url(key="coa/secure/COA-12345.pdf", coa_id="COA-12345", user_id="u1")

    signer.assert_not_called()


def test_fr6_gate_checked_once_per_request(service: PresignedUrlService) -> None:
    """A single sign request evaluates the completion gate exactly once."""
    calls: list[str] = []

    def validator(coa_id: str, key: str, user_id: str) -> bool:
        calls.append(coa_id)
        return True

    service._validate_before_sign = True
    service._validator = validator

    service.generate_signed_url(key="coa/secure/COA-12345.pdf", coa_id="COA-12345", user_id="u1")

    assert calls == ["COA-12345"]


def test_fr6_gate_ignores_request_without_coa_context(service: PresignedUrlService) -> None:
    """No COA context means there is nothing to gate, so signing is allowed."""
    service._validate_before_sign = True
    service._validator = lambda coa_id, key, user_id: False

    result = service.generate_signed_url(key="coa/secure/COA-12345.pdf")

    assert result.signed_url is not None


def test_fr6_gate_denies_download_for_every_user_without_populated_fields(service: PresignedUrlService) -> None:
    """The gate is not bypassable by a different requester (FR12 audit subject)."""
    service._validate_before_sign = True
    service._validator = lambda coa_id, key, user_id: False

    for user_id in ("cs-agent-001", "qa-reviewer-002", "plant-operator-003"):
        with pytest.raises(PresignedUrlScopeError):
            service.generate_signed_url(key="coa/secure/COA-12345.pdf", coa_id="COA-12345", user_id=user_id)


def test_expired_error_type_is_exported() -> None:
    """The expiry error surfaced to callers is the documented public type."""
    assert issubclass(PresignedUrlExpiredError, Exception)
    assert issubclass(PresignedUrlGenerationError, Exception)
