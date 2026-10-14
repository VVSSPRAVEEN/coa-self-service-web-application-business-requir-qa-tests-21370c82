"""FR1: OIDC login - token verification against the identity provider.

These tests drive the real :mod:`src.auth.oidc` verification path with a token
built by hand, so no network and no provider are involved. The symmetric mode
under test (``AUTH_HS256_SECRET``) is the one the deployment uses when it mints
its own tokens; the JWKS mode is a different code path that needs a live key
document.

The HMAC key material used here is generated afresh for every run (see
:func:`_random_signing_material`) and injected through ``monkeypatch``, so no
credential is ever committed to the repository while the signature path itself
is still exercised for real.

Every test pins the whole provider configuration through ``monkeypatch`` - the
key material *and* the issuer/audience - because a stale ``COGNITO_*`` variable
in the environment would otherwise silently change which checks run.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import secrets
import time
from collections.abc import Iterator
from typing import Any

import pytest

from src.auth.oidc import (
    AuthResponse,
    TokenValidationError,
    User,
    get_provider_config,
    login_flow,
    signature_verification_available,
    validate_token,
)


def _random_signing_material() -> str:
    """Return a fresh, unpredictable HMAC key for the symmetric provider mode."""
    return secrets.token_hex(32)


def _unrelated_signing_material(current: str) -> str:
    """Return key material that is guaranteed to differ from ``current``."""
    candidate = _random_signing_material()
    while candidate == current:
        candidate = _random_signing_material()
    return candidate


def _b64(raw: bytes) -> str:
    return base64.urlsafe_b64encode(raw).decode("ascii").rstrip("=")


def _token(claims: dict[str, Any], material: str, alg: str = "HS256") -> str:
    """Build a compact JWS the way the provider would."""
    header = _b64(json.dumps({"alg": alg, "typ": "JWT"}).encode("utf-8"))
    payload = _b64(json.dumps(claims).encode("utf-8"))
    signing_input = f"{header}.{payload}"
    signature = _b64(
        hmac.new(material.encode("utf-8"), signing_input.encode("ascii"), hashlib.sha256).digest()
    )
    return f"{signing_input}.{signature}"


def _claims(**overrides: Any) -> dict[str, Any]:
    now = int(time.time())
    claims: dict[str, Any] = {
        "sub": "user-4711",
        "preferred_username": "dana.ops",
        "email": "dana.ops@example.com",
        "cognito:groups": ["ventura-internal"],
        "custom:plants": ["PLANT-A", "PLANT-B"],
        "iat": now,
        "exp": now + 900,
    }
    claims.update(overrides)
    return claims


@pytest.fixture(autouse=True)
def provider_env(monkeypatch: pytest.MonkeyPatch) -> Iterator[str]:
    """Pin the symmetric provider mode and clear every JWKS-mode variable."""
    material = _random_signing_material()
    monkeypatch.setenv("AUTH_HS256_SECRET", material)
    monkeypatch.delenv("COGNITO_JWKS_URL", raising=False)
    monkeypatch.delenv("COGNITO_ISSUER", raising=False)
    monkeypatch.delenv("COGNITO_CLIENT_ID", raising=False)
    yield material


@pytest.fixture
def signing_material(provider_env: str) -> str:
    """The HMAC material ``provider_env`` installed for this test."""
    return provider_env


class TestValidateToken:
    def test_a_correctly_signed_token_yields_the_normalised_user(self, signing_material: str) -> None:
        user = validate_token(_token(_claims(), signing_material))

        assert isinstance(user, User)
        assert user.user_id == "user-4711"
        assert user.username == "dana.ops"
        assert user.email == "dana.ops@example.com"

    def test_roles_are_mapped_from_the_cognito_groups_claim(self, signing_material: str) -> None:
        user = validate_token(
            _token(_claims(**{"cognito:groups": ["3pl", "customer"]}), signing_material)
        )

        assert user.roles == ["3pl", "customer"]

    def test_plants_are_mapped_from_the_custom_claim(self, signing_material: str) -> None:
        assert validate_token(_token(_claims(), signing_material)).plants == ["PLANT-A", "PLANT-B"]

    def test_a_scalar_group_claim_is_normalised_to_a_list(self, signing_material: str) -> None:
        user = validate_token(_token(_claims(**{"cognito:groups": "customer"}), signing_material))

        assert user.roles == ["customer"]

    def test_a_roles_claim_is_the_fallback_when_groups_is_absent(self, signing_material: str) -> None:
        claims = _claims(roles=["internal"])
        del claims["cognito:groups"]

        assert validate_token(_token(claims, signing_material)).roles == ["internal"]

    def test_the_groups_claim_wins_over_the_roles_claim(self, signing_material: str) -> None:
        user = validate_token(
            _token(_claims(roles=["internal"], **{"cognito:groups": ["3pl"]}), signing_material)
        )

        assert user.roles == ["3pl"]

    def test_a_token_without_groups_or_roles_has_no_roles(self, signing_material: str) -> None:
        claims = _claims()
        del claims["cognito:groups"]

        assert validate_token(_token(claims, signing_material)).roles == []

    def test_the_username_falls_back_through_the_claim_precedence(self, signing_material: str) -> None:
        claims = _claims()
        del claims["preferred_username"]
        claims["cognito:username"] = "fallback.name"

        assert validate_token(_token(claims, signing_material)).username == "fallback.name"


class TestRejectedTokens:
    def test_an_expired_token_is_rejected(self, signing_material: str) -> None:
        with pytest.raises(TokenValidationError):
            validate_token(_token(_claims(exp=int(time.time()) - 3600), signing_material))

    def test_a_token_without_an_expiry_is_rejected(self, signing_material: str) -> None:
        claims = _claims()
        del claims["exp"]

        with pytest.raises(TokenValidationError):
            validate_token(_token(claims, signing_material))

    def test_a_token_that_is_not_yet_valid_is_rejected(self, signing_material: str) -> None:
        with pytest.raises(TokenValidationError):
            validate_token(_token(_claims(nbf=int(time.time()) + 3600), signing_material))

    def test_a_token_signed_with_another_key_is_rejected(self, signing_material: str) -> None:
        other = _unrelated_signing_material(signing_material)

        with pytest.raises(TokenValidationError):
            validate_token(_token(_claims(), other))

    def test_a_tampered_payload_is_rejected(self, signing_material: str) -> None:
        header = _b64(json.dumps({"alg": "HS256", "typ": "JWT"}).encode("utf-8"))
        payload = _b64(json.dumps(_claims(**{"cognito:groups": ["internal"]})).encode("utf-8"))
        signature = _b64(
            hmac.new(
                signing_material.encode("utf-8"),
                f"{header}.{payload}".encode("ascii"),
                hashlib.sha256,
            ).digest()
        )

        with pytest.raises(TokenValidationError):
            validate_token(f"{header}.{payload}.{signature[:-4]}AAAA")

    def test_the_alg_none_token_is_rejected(self, signing_material: str) -> None:
        header = _b64(json.dumps({"alg": "none", "typ": "JWT"}).encode("utf-8"))
        payload = _b64(json.dumps(_claims()).encode("utf-8"))

        with pytest.raises(TokenValidationError):
            validate_token(f"{header}.{payload}.")

    def test_a_non_hs256_algorithm_is_rejected(self, signing_material: str) -> None:
        with pytest.raises(TokenValidationError):
            validate_token(_token(_claims(), signing_material, alg="RS256"))

    @pytest.mark.parametrize(
        "malformed",
        [
            pytest.param("not-a-jwt", id="not_a_jwt"),
            pytest.param("only.two", id="two_segments"),
            pytest.param("a.b.c.d", id="four_segments"),
            pytest.param(".payload.sig", id="empty_header"),
            pytest.param("header..sig", id="empty_payload"),
            pytest.param("", id="empty_token"),
            pytest.param("   ", id="blank_token"),
        ],
    )
    def test_a_malformed_token_is_rejected(self, malformed: str) -> None:
        with pytest.raises(TokenValidationError):
            validate_token(malformed)

    def test_a_token_without_a_subject_is_rejected(self, signing_material: str) -> None:
        claims = _claims()
        del claims["sub"]

        with pytest.raises(TokenValidationError):
            validate_token(_token(claims, signing_material))

    def test_a_non_numeric_expiry_is_rejected(self, signing_material: str) -> None:
        with pytest.raises(TokenValidationError):
            validate_token(_token(_claims(exp="soon"), signing_material))


class TestAudienceAndIssuer:
    def test_a_mismatched_audience_is_rejected(
        self, monkeypatch: pytest.MonkeyPatch, signing_material: str
    ) -> None:
        monkeypatch.setenv("COGNITO_CLIENT_ID", "ventura-portal")

        with pytest.raises(TokenValidationError):
            validate_token(_token(_claims(aud="some-other-app"), signing_material))

    def test_a_matching_audience_is_accepted(
        self, monkeypatch: pytest.MonkeyPatch, signing_material: str
    ) -> None:
        monkeypatch.setenv("COGNITO_CLIENT_ID", "ventura-portal")

        assert validate_token(_token(_claims(aud="ventura-portal"), signing_material)).user_id == "user-4711"

    def test_a_mismatched_issuer_is_rejected(
        self, monkeypatch: pytest.MonkeyPatch, signing_material: str
    ) -> None:
        monkeypatch.setenv("COGNITO_ISSUER", "https://issuer.example")

        with pytest.raises(TokenValidationError):
            validate_token(_token(_claims(iss="https://evil.example"), signing_material))


class TestLoginFlow:
    def test_the_exchange_returns_the_session_identity_and_expiry(self, signing_material: str) -> None:
        expires = int(time.time()) + 900

        result = login_flow(_token(_claims(exp=expires), signing_material))

        assert isinstance(result, AuthResponse)
        assert result.user.user_id == "user-4711"
        assert result.user.roles == ["ventura-internal"]
        assert result.expires_at.startswith("20")

    def test_the_expiry_comes_from_the_token_not_the_clock(self, signing_material: str) -> None:
        result = login_flow(_token(_claims(exp=4102444800), signing_material))

        assert result.expires_at == "2100-01-01T00:00:00+00:00"

    def test_a_rejected_token_raises_rather_than_returning_a_session(self, signing_material: str) -> None:
        with pytest.raises(TokenValidationError):
            login_flow(_token(_claims(exp=int(time.time()) - 3600), signing_material))


class TestProviderConfiguration:
    def test_hs256_mode_is_reported_and_can_verify(self, signing_material: str) -> None:
        config = get_provider_config()

        assert config["signature_mode"] == "hs256"
        assert signature_verification_available() is True

    def test_the_signing_material_is_never_returned(self, signing_material: str) -> None:
        assert signing_material not in json.dumps(get_provider_config())

    def test_no_configuration_at_all_reports_that_signatures_cannot_be_verified(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("AUTH_HS256_SECRET", raising=False)

        assert get_provider_config()["signature_mode"] == "none"
        assert signature_verification_available() is False
