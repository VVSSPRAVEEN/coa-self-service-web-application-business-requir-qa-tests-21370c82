"""Integration tests for the eight approved contract paths.

These tests drive the real FastAPI application through
``fastapi.testclient.TestClient`` - the same ``app`` object the container serves -
and stay hermetic:

* the Mule COA service is replaced through FastAPI's ``dependency_overrides``
  with a fake that returns two in-memory ``COARecord`` objects, so no HTTP call
  leaves the process;
* the presigned-URL factory is replaced with a stub that returns a real
  ``PresignedUrlResult`` (one test deliberately leaves the real factory in place
  to prove the credential-less 503 branch);
* the audit sink is replaced with a recorder, so no trail write is attempted - and
  two tests replace it with a sink that *raises*, to prove the distribution is
  refused rather than issued unaudited;
* the print service is built with a test secret so a re-auth token can be minted
  by the owning module itself;
* ``ADMIN_USERS_PATH`` points at a temporary file.

One test signs an HS256 token by hand with :mod:`hmac`, exercising the real
``src.auth.oidc`` verification path with no network and no external provider.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import os
import time
from datetime import datetime, timedelta, timezone
from typing import Any, Iterator, Optional

import pytest
from fastapi.testclient import TestClient

from src.api import contract_api
from src.api.mule.coa_service import COARecord, CompletionStatus, MuleApiError, get_coa_service
from src.api.mule.endpoints import app
from src.documents.print import PrintService
from src.documents.s3_service.presigned_url_service import PresignedUrlResult
HS256_SECRET = os.environ.get("CONTRACT_TEST_HS256_SECRET") or os.urandom(48).hex()
PRINT_SECRET = b"contract-test-print-reauth-secret"

PASSING_RESULTS = {
    "missing": [],
    "fields": {
        "moisture": {"status": "pass", "value": "12.1%"},
        "protein": {"status": "pass", "value": "41.0%"},
    },
}
INCOMPLETE_RESULTS = {
    "missing": ["moisture", "protein"],
    "fields": {"moisture": {"status": "pending", "value": None}, "protein": None},
}

CONTRACT_ROUTES = {
    ("POST", "/auth/login"),
    ("GET", "/coas"),
    ("GET", "/coas/{coa_id}"),
    ("POST", "/coas/{coa_id}/download"),
    ("POST", "/coas/{coa_id}/print"),
    ("POST", "/coas/{coa_id}/print/reauth"),
    ("POST", "/coas/{coa_id}/validate"),
    ("GET", "/missing-coas"),
    ("POST", "/admin/users"),
}


def _complete_record() -> COARecord:
    return COARecord(
        coa_number="COA-1001",
        customer_id="CUST-77",
        plant_code="PLANT-A",
        lot_number="LOT-11",
        product_code="PROD-9",
        issue_date="2026-03-01",
        completion_status=CompletionStatus.COMPLETED,
        status="completed",
        file_size_bytes=20481,
        test_results=PASSING_RESULTS,
        generated_at=datetime(2026, 3, 1, 12, 0, tzinfo=timezone.utc),
        completed_at="2026-03-01T12:05:00Z",
    )


def _incomplete_record() -> COARecord:
    return COARecord(
        coa_number="COA-1002",
        customer_id="CUST-88",
        plant_code="PLANT-B",
        lot_number="LOT-22",
        product_code="PROD-4",
        issue_date="2026-02-01",
        completion_status=CompletionStatus.PENDING,
        status="pending",
        file_size_bytes=None,
        test_results=INCOMPLETE_RESULTS,
    )


class FakeCoaService:
    """In-memory COA service: records the filters it was asked for."""

    def __init__(self, records: list[COARecord]) -> None:
        self.records = list(records)
        self.by_number = {record.coa_number: record for record in self.records}
        self.search_filters: list[Any] = []
        self.error: Optional[Exception] = None

    async def search_coa(self, filters: Any) -> list[COARecord]:
        self.search_filters.append(filters)
        if self.error is not None:
            raise self.error
        return list(self.records)

    async def get_coa_by_number(self, coa_number: str) -> Optional[COARecord]:
        if self.error is not None:
            raise self.error
        return self.by_number.get(coa_number)


class FakePresignedService:
    """Presigned-URL stub returning the real result type."""

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def generate_signed_url(
        self,
        key: str,
        method: str = "GET",
        expiry_seconds: Optional[int] = None,
        user_id: Optional[str] = None,
        coa_id: Optional[str] = None,
    ) -> PresignedUrlResult:
        lifetime = expiry_seconds or 900
        self.calls.append(
            {
                "key": key,
                "method": method,
                "expiry_seconds": expiry_seconds,
                "user_id": user_id,
                "coa_id": coa_id,
            }
        )
        return PresignedUrlResult(
            signed_url=(
                "https://coa-documents.s3.us-east-1.amazonaws.com/"
                f"{key}?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Signature=stub"
            ),
            bucket="coa-documents",
            key=key,
            method=method.upper(),
            expires_at=datetime.now(timezone.utc) + timedelta(seconds=lifetime),
            request_id="stub-request-id",
            scope_hash="stub-scope-hash",
        )


@pytest.fixture
def coa_service() -> FakeCoaService:
    return FakeCoaService([_complete_record(), _incomplete_record()])


@pytest.fixture
def presigned() -> FakePresignedService:
    return FakePresignedService()


@pytest.fixture
def audit_log() -> list[tuple[str, dict[str, Any]]]:
    return []


@pytest.fixture
def print_service() -> PrintService:
    return PrintService(reauth_secret=PRINT_SECRET)


@pytest.fixture
def client(
    coa_service: FakeCoaService,
    presigned: FakePresignedService,
    audit_log: list[tuple[str, dict[str, Any]]],
    print_service: PrintService,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> Iterator[TestClient]:
    """A TestClient with every external boundary replaced by an in-process double."""
    monkeypatch.setenv("ADMIN_USERS_PATH", str(tmp_path / "admin_users.json"))
    monkeypatch.setenv("AUTH_HS256_SECRET", HS256_SECRET)
    monkeypatch.delenv("COGNITO_ISSUER", raising=False)
    monkeypatch.delenv("COGNITO_CLIENT_ID", raising=False)
    monkeypatch.delenv("COGNITO_JWKS_URL", raising=False)
    monkeypatch.delenv("PRINT_REAUTH_SECRET", raising=False)
    monkeypatch.delenv("JDE_BASE_URL", raising=False)
    for variable in (
        "AWS_ACCESS_KEY_ID",
        "AWS_SECRET_ACCESS_KEY",
        "AWS_SESSION_TOKEN",
    ):
        monkeypatch.delenv(variable, raising=False)

    # No JDE integration in this suite: the gate answers from the record, and the
    # live-JDE path is covered by tests/unit/test_jde_integration.py.
    monkeypatch.setattr(contract_api, "build_jde_client", lambda: None)

    monkeypatch.setattr(
        contract_api, "build_presigned_url_service", lambda **kwargs: presigned
    )
    monkeypatch.setattr(contract_api, "build_print_service", lambda: print_service)
    monkeypatch.setattr(
        contract_api,
        "log_access",
        lambda actor, action, resource, metadata=None: audit_log.append(
            ("access", {"actor": actor, "action": action, "resource": resource, "metadata": metadata})
        ),
    )
    monkeypatch.setattr(
        contract_api,
        "log_generation",
        lambda actor, resource, action, metadata=None: audit_log.append(
            ("generation", {"actor": actor, "resource": resource, "action": action, "metadata": metadata})
        ),
    )

    app.dependency_overrides[get_coa_service] = lambda: coa_service
    with TestClient(app) as test_client:
        yield test_client
    app.dependency_overrides.clear()


def _b64(raw: bytes) -> str:
    return base64.urlsafe_b64encode(raw).decode("ascii").rstrip("=")


def _mounted_routes(application: Any) -> set[tuple[str, str]]:
    """Every ``(method, path)`` the application serves.

    FastAPI 0.142/starlette 1.x keeps an included router as a single
    ``_IncludedRouter`` entry in ``app.routes`` instead of copying its routes, so
    the walk follows ``original_router`` rather than assuming a flat list.
    """
    found: set[tuple[str, str]] = set()
    pending = list(application.routes)
    while pending:
        route = pending.pop()
        path = getattr(route, "path", None)
        methods = getattr(route, "methods", None)
        if path and methods:
            found.update((method, path) for method in methods)
        original = getattr(route, "original_router", None)
        if original is not None:
            pending.extend(getattr(original, "routes", []))
    return found


def _hs256_token(claims: dict[str, Any], secret: str = HS256_SECRET) -> str:
    header = _b64(json.dumps({"alg": "HS256", "typ": "JWT"}).encode("utf-8"))
    payload = _b64(json.dumps(claims).encode("utf-8"))
    signing_input = f"{header}.{payload}"
    signature = _b64(
        hmac.new(secret.encode("utf-8"), signing_input.encode("ascii"), hashlib.sha256).digest()
    )
    return f"{signing_input}.{signature}"


def _reauth_token(user_id: str, coa_id: str) -> str:
    """Mint the FR8 re-auth credential that ``PrintService`` verifies.

    This helper follows the verification contract in
    ``PrintService._verify_reauth`` by hand - including the base64 signature
    padding that ``_compute_signature`` emits - so the print tests exercise the
    verifier against a token this file produced, not against one the service
    minted for itself. :class:`TestPrintReauth` covers the other direction, where
    the real ``PrintService.issue_reauth_token`` mints and the endpoint returns
    it, so the two together pin both halves of the format.
    """
    issued_at = int(time.time())
    header = _b64(json.dumps({"alg": "HS256", "typ": "JWT"}).encode("utf-8"))
    payload = _b64(
        json.dumps({"user_id": user_id, "coa_id": coa_id, "iat": issued_at}).encode("utf-8")
    )
    digest = hmac.new(
        PRINT_SECRET,
        f"{user_id}|{coa_id}|{issued_at}".encode("utf-8"),
        hashlib.sha256,
    ).digest()
    signature = base64.urlsafe_b64encode(digest).decode("ascii")
    return f"{header}.{payload}.{signature}"


def _login_claims(**overrides: Any) -> dict[str, Any]:
    claims = {
        "sub": "user-4711",
        "preferred_username": "dana.ops",
        "email": "dana.ops@example.com",
        "cognito:groups": ["ventura-internal"],
        "custom:plants": ["PLANT-A"],
        "exp": int(time.time()) + 900,
        "iat": int(time.time()),
    }
    claims.update(overrides)
    return claims


class TestRouteTable:
    def test_the_application_exposes_the_eight_contract_paths(self) -> None:
        assert CONTRACT_ROUTES <= _mounted_routes(app)

    def test_the_contract_paths_appear_in_the_openapi_schema(self) -> None:
        schema = app.openapi()
        declared = {
            (method.upper(), path)
            for path, operations in schema["paths"].items()
            for method in operations
        }

        assert CONTRACT_ROUTES <= declared

    def test_the_legacy_coa_router_is_still_mounted(self) -> None:
        mounted = _mounted_routes(app)

        assert ("GET", "/coa/search") in mounted
        assert ("GET", "/health") in mounted


class TestListCoas:
    def test_returns_the_contract_dto_for_every_record(self, client: TestClient) -> None:
        response = client.get("/coas")

        assert response.status_code == 200
        body = response.json()
        assert body["page"] == 1
        assert body["page_size"] == 25
        assert body["total"] == 2
        assert body["total_pages"] == 1

        first = body["items"][0]
        assert first["coa_id"] == "COA-1001"
        assert first["plant_of_origin"] == "PLANT-A"
        assert first["completion_status"] == "completed"
        assert first["customer_id"] == "CUST-77"
        assert first["lot_number"] == "LOT-11"
        assert first["product_code"] == "PROD-9"
        assert first["issue_date"] == "2026-03-01"
        assert first["file_size_bytes"] == 20481
        assert first["status"] == "completed"
        assert first["generated_at"] == "2026-03-01T12:00:00+00:00"
        assert first["completed_at"] == "2026-03-01T12:05:00Z"
        assert body["items"][1]["completion_status"] == "pending"

    def test_maps_the_query_parameters_onto_the_service_filter(
        self, client: TestClient, coa_service: FakeCoaService
    ) -> None:
        response = client.get(
            "/coas",
            params={
                "search": "COA-1001",
                "status": "complete",
                "date_from": "2026-01-01",
                "date_to": "2026-03-31",
                "page": 1,
                "page_size": 1,
            },
        )

        assert response.status_code == 200
        body = response.json()
        assert len(body["items"]) == 1
        assert body["page_size"] == 1
        assert body["total"] == 2

        applied = coa_service.search_filters[-1]
        assert applied.q == "COA-1001"
        assert applied.status == "completed"
        assert applied.start_date == "2026-01-01"
        assert applied.end_date == "2026-03-31"

    def test_rejects_an_out_of_range_page_size(self, client: TestClient) -> None:
        assert client.get("/coas", params={"page_size": 5000}).status_code == 422

    def test_maps_a_mule_failure_to_503(
        self, client: TestClient, coa_service: FakeCoaService
    ) -> None:
        coa_service.error = MuleApiError(code="mule_down", message="mule is unreachable")

        response = client.get("/coas")

        assert response.status_code == 503
        assert response.json()["detail"]["code"] == "mule_down"

    def test_plant_is_passed_through_to_the_service_filter(
        self, client: TestClient, coa_service: FakeCoaService
    ) -> None:
        response = client.get("/coas", params={"plant": "PLANT-A"})

        assert response.status_code == 200
        assert coa_service.search_filters[-1].plant == "PLANT-A"

    def test_product_filters_the_records_server_side(
        self, client: TestClient, coa_service: FakeCoaService
    ) -> None:
        response = client.get("/coas", params={"product": "PROD-9"})

        assert response.status_code == 200
        body = response.json()
        assert [item["coa_id"] for item in body["items"]] == ["COA-1001"]
        assert body["total"] == 1

    def test_the_product_filter_counts_only_matching_records_before_pagination(
        self, client: TestClient, coa_service: FakeCoaService
    ) -> None:
        response = client.get("/coas", params={"product": "PROD-4", "page_size": 1})

        assert response.status_code == 200
        body = response.json()
        assert body["total"] == 1
        assert body["total_pages"] == 1
        assert body["items"][0]["product_code"] == "PROD-4"

    def test_an_unknown_product_matches_nothing(
        self, client: TestClient, coa_service: FakeCoaService
    ) -> None:
        response = client.get("/coas", params={"product": "PROD-NOT-THERE"})

        assert response.status_code == 200
        assert response.json() == {
            "items": [],
            "page": 1,
            "page_size": 25,
            "total": 0,
            "total_pages": 0,
        }

    def test_plant_and_product_are_both_accepted_together(
        self, client: TestClient, coa_service: FakeCoaService
    ) -> None:
        response = client.get("/coas", params={"plant": "PLANT-A", "product": "PROD-9"})

        assert response.status_code == 200
        assert coa_service.search_filters[-1].plant == "PLANT-A"
        assert [item["coa_id"] for item in response.json()["items"]] == ["COA-1001"]

    def test_the_product_filter_is_case_insensitive(
        self, client: TestClient, coa_service: FakeCoaService
    ) -> None:
        response = client.get("/coas", params={"product": "prod-9"})

        assert [item["coa_id"] for item in response.json()["items"]] == ["COA-1001"]


class TestGetCoaDetail:
    def test_returns_the_contract_dto(self, client: TestClient) -> None:
        response = client.get("/coas/COA-1001")

        assert response.status_code == 200
        body = response.json()
        assert body["coa_id"] == "COA-1001"
        assert body["plant_of_origin"] == "PLANT-A"
        assert body["completion_status"] == "completed"

    def test_reports_an_unknown_coa_as_404(self, client: TestClient) -> None:
        response = client.get("/coas/COA-9999")

        assert response.status_code == 404
        assert response.json()["detail"]["coa_id"] == "COA-9999"

    def test_maps_a_mule_failure_to_503(
        self, client: TestClient, coa_service: FakeCoaService
    ) -> None:
        coa_service.error = MuleApiError(code="mule_down", message="mule is unreachable")

        response = client.get("/coas/COA-1001")

        assert response.status_code == 503
        assert response.json()["detail"]["code"] == "mule_down"


class TestValidateCoa:
    def test_reports_a_complete_coa_as_releasable(self, client: TestClient) -> None:
        response = client.post("/coas/COA-1001/validate")

        assert response.status_code == 200
        assert response.json() == {
            "coa_id": "COA-1001",
            "completion_status": "complete",
            "releasable": True,
            "missing_fields": [],
        }

    def test_reports_an_incomplete_coa_as_blocked(self, client: TestClient) -> None:
        response = client.post("/coas/COA-1002/validate")

        assert response.status_code == 200
        body = response.json()
        assert body["coa_id"] == "COA-1002"
        assert body["completion_status"] == "incomplete"
        assert body["releasable"] is False
        assert body["missing_fields"] == ["moisture", "protein"]

    def test_reports_an_unknown_coa_as_404(self, client: TestClient) -> None:
        assert client.post("/coas/COA-9999/validate").status_code == 404


class TestDownloadCoa:
    def test_an_incomplete_coa_is_blocked_with_its_missing_fields(
        self, client: TestClient
    ) -> None:
        response = client.post("/coas/COA-1002/download")

        assert response.status_code == 409
        body = response.json()
        assert body["code"] == "completion_gate_failed"
        assert body["coa_id"] == "COA-1002"
        assert body["missing_fields"] == ["moisture", "protein"]
        assert "detail" in body

    def test_a_complete_coa_receives_a_presigned_url(
        self, client: TestClient, presigned: FakePresignedService, audit_log: list
    ) -> None:
        response = client.post(
            "/coas/COA-1001/download",
            headers={"X-User-Id": "user-4711"},
            json={"expiry_seconds": 600},
        )

        assert response.status_code == 200
        body = response.json()
        assert body["coa_id"] == "COA-1001"
        assert body["url"].startswith("https://coa-documents.s3.us-east-1.amazonaws.com/")
        assert body["method"] == "GET"
        assert body["expires_in"] == 600
        assert body["audit_recorded"] is True

        assert presigned.calls[-1]["key"] == "coa/secure/coas/COA-1001.pdf"
        assert presigned.calls[-1]["coa_id"] == "COA-1001"
        assert presigned.calls[-1]["expiry_seconds"] == 600
        assert audit_log[-1][0] == "access"
        assert audit_log[-1][1]["actor"] == "user-4711"

    def test_the_body_is_optional_and_defaults_to_the_ceiling(
        self, client: TestClient, presigned: FakePresignedService
    ) -> None:
        response = client.post("/coas/COA-1001/download")

        assert response.status_code == 200
        assert response.json()["expires_in"] == 900
        assert presigned.calls[-1]["expiry_seconds"] == 900

    def test_rejects_an_expiry_beyond_the_ceiling(self, client: TestClient) -> None:
        response = client.post("/coas/COA-1001/download", json={"expiry_seconds": 901})

        assert response.status_code == 422

    def test_an_unknown_coa_is_a_404(self, client: TestClient) -> None:
        assert client.post("/coas/COA-9999/download").status_code == 404

    def test_without_signing_credentials_the_real_factory_reports_503(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        service = FakeCoaService([_complete_record()])
        presigned = FakePresignedService()
        audit_log: list = []
        for variable in ("AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", "AWS_SESSION_TOKEN"):
            monkeypatch.delenv(variable, raising=False)
        monkeypatch.setattr(
            contract_api,
            "log_access",
            lambda actor, action, resource, metadata=None: audit_log.append(
                ("access", {"actor": actor})
            ),
        )
        app.dependency_overrides[get_coa_service] = lambda: service
        try:
            with TestClient(app) as real_client:
                response = real_client.post("/coas/COA-1001/download")
        finally:
            app.dependency_overrides.clear()

        assert presigned.calls == []
        assert response.status_code == 503
        assert response.json()["code"] == "presigned_url_unavailable"


class TestDownloadBlocksWhenTheAuditTrailCannotRecordIt:
    """FR12: a COA may not leave the platform if the trail never saw it release.

    These are the tests the fail-open implementation could not have written: the
    signed URL is the whole point of the endpoint, so a sink that raises used to
    produce a 200 whose ``audit_recorded`` was ``false``. The distribution is now
    refused with ``503 audit_write_failed`` and no grant in the body.
    """

    def test_a_failing_audit_sink_refuses_the_download(
        self, client: TestClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def exploding_sink(actor: str, action: str, resource: str, metadata: Any = None) -> None:
            raise MuleApiError(
                code="audit_write_failed", message="the audit trail is unavailable"
            )

        monkeypatch.setattr(contract_api, "log_access", exploding_sink)

        response = client.post("/coas/COA-1001/download", headers={"X-User-Id": "user-4711"})

        assert response.status_code == 503
        body = response.json()
        assert body["code"] == "audit_write_failed"
        assert body["coa_id"] == "COA-1001"
        assert "url" not in body
        assert "scope_id" not in body
        assert "audit_recorded" not in body

    def test_an_unexpected_audit_error_is_also_blocking(
        self, client: TestClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def exploding_sink(**kwargs: Any) -> None:
            raise RuntimeError("the trail connection pool is exhausted")

        monkeypatch.setattr(contract_api, "log_access", exploding_sink)

        response = client.post("/coas/COA-1001/download")

        assert response.status_code == 503
        assert response.json()["code"] == "audit_write_failed"

    def test_the_single_use_scope_is_retired_when_the_audit_fails(
        self, client: TestClient, monkeypatch: pytest.MonkeyPatch, tmp_path
    ) -> None:
        """The signed URL is never returned, so nothing may be left consumable."""
        from src.documents.s3_service.consumption import SingleUseUrlStore

        store = SingleUseUrlStore()
        monkeypatch.setattr(contract_api, "get_consumption_store", lambda: store)

        def exploding_sink(**kwargs: Any) -> None:
            raise MuleApiError(code="audit_write_failed", message="sink down")

        monkeypatch.setattr(contract_api, "log_access", exploding_sink)

        response = client.post("/coas/COA-1001/download", headers={"X-User-Id": "user-4711"})

        assert response.status_code == 503
        assert store.active_scopes() == {}

    def test_the_gate_still_runs_before_the_audit_write(
        self, client: TestClient, monkeypatch: pytest.MonkeyPatch, audit_log: list
    ) -> None:
        """An incomplete COA is refused by the gate; the sink is never consulted."""
        monkeypatch.setattr(
            contract_api,
            "log_access",
            lambda **kwargs: (_ for _ in ()).throw(
                AssertionError("the audit sink must not be reached for a blocked COA")
            ),
        )

        response = client.post("/coas/COA-1002/download")

        assert response.status_code == 409
        assert response.json()["code"] == "completion_gate_failed"

    def test_a_successful_download_still_reports_the_audit_as_recorded(
        self, client: TestClient
    ) -> None:
        response = client.post("/coas/COA-1001/download")

        assert response.status_code == 200
        assert response.json()["audit_recorded"] is True


class TestPrintCoa:
    def test_a_valid_reauth_token_is_accepted(
        self,
        client: TestClient,
        audit_log: list,
    ) -> None:
        token = _reauth_token("user-4711", "COA-1001")

        response = client.post(
            "/coas/COA-1001/print",
            json={"user_id": "user-4711", "reauth_token": token},
        )

        assert response.status_code == 200
        body = response.json()
        assert body["coa_id"] == "COA-1001"
        assert body["success"] is True
        assert body["audit_event"]["action"] == "print_success"
        assert body["audit_recorded"] is True
        assert audit_log[-1][0] == "generation"

    def test_an_invalid_reauth_token_is_a_401(self, client: TestClient) -> None:
        response = client.post(
            "/coas/COA-1001/print",
            json={"user_id": "user-4711", "reauth_token": "not-a-token"},
        )

        assert response.status_code == 401
        assert response.json()["code"] == "print_reauth_failed"

    def test_a_reauth_token_for_another_coa_is_a_401(self, client: TestClient) -> None:
        token = _reauth_token("user-4711", "COA-1002")

        response = client.post(
            "/coas/COA-1001/print",
            json={"user_id": "user-4711", "reauth_token": token},
        )

        assert response.status_code == 401

    def test_an_incomplete_coa_is_blocked_before_reauth(self, client: TestClient) -> None:
        response = client.post(
            "/coas/COA-1002/print",
            json={"user_id": "user-4711", "reauth_token": "irrelevant"},
        )

        assert response.status_code == 409
        assert response.json()["code"] == "completion_gate_failed"

    def test_a_cross_plant_print_without_a_mapping_is_denied(self, client: TestClient) -> None:
        token = _reauth_token("user-4711", "COA-1001")

        response = client.post(
            "/coas/COA-1001/print",
            json={
                "user_id": "user-4711",
                "reauth_token": token,
                "printing_plant": "PLANT-Z",
            },
        )

        assert response.status_code == 403
        assert response.json()["code"] == "sister_plant_mapping_missing"

    def test_a_same_plant_print_is_not_subject_to_the_cross_plant_check(
        self, client: TestClient
    ) -> None:
        token = _reauth_token("user-4711", "COA-1001")

        response = client.post(
            "/coas/COA-1001/print",
            json={
                "user_id": "user-4711",
                "reauth_token": token,
                "printing_plant": "PLANT-A",
            },
        )

        assert response.status_code == 200

    def test_an_unconfigured_reauth_service_is_a_401(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from src.documents.print.print_service import ReauthError

        def _boom() -> PrintService:
            raise ReauthError("PRINT_REAUTH_SECRET is not configured")

        service = FakeCoaService([_complete_record()])
        monkeypatch.setattr(contract_api, "build_print_service", _boom)
        app.dependency_overrides[get_coa_service] = lambda: service
        try:
            with TestClient(app) as real_client:
                response = real_client.post(
                    "/coas/COA-1001/print",
                    json={"user_id": "user-4711", "reauth_token": "any"},
                )
        finally:
            app.dependency_overrides.clear()

        assert response.status_code == 401
        assert response.json()["code"] == "reauth_unavailable"

    def test_the_body_is_required(self, client: TestClient) -> None:
        assert client.post("/coas/COA-1001/print", json={}).status_code == 422

    def test_a_failing_audit_sink_refuses_the_print(
        self, client: TestClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def exploding_sink(actor: str, resource: str, action: str, metadata: Any = None) -> None:
            raise MuleApiError(
                code="audit_write_failed", message="the audit trail is unavailable"
            )

        monkeypatch.setattr(contract_api, "log_generation", exploding_sink)

        response = client.post(
            "/coas/COA-1001/print",
            json={"user_id": "user-4711", "reauth_token": _reauth_token("user-4711", "COA-1001")},
        )

        assert response.status_code == 503
        body = response.json()
        assert body["code"] == "audit_write_failed"
        assert body["coa_id"] == "COA-1001"
        assert "presigned_url" not in body
        assert "success" not in body


class TestMissingCoas:
    def test_lists_the_outstanding_coas(self, client: TestClient) -> None:
        response = client.get("/missing-coas")

        assert response.status_code == 200
        body = response.json()
        assert body["generated_at"]
        assert [item["coa_id"] for item in body["items"]] == ["COA-1002"]

        item = body["items"][0]
        assert set(item) == {
            "coa_id",
            "vessel",
            "voyage",
            "counterparty",
            "priority",
            "required_ship_date",
            "age_days",
            "blocking_reason",
        }
        assert item["counterparty"] == "CUST-88"
        assert item["required_ship_date"] == "2026-02-01"
        assert item["age_days"] >= 0
        assert "missing priority" in item["blocking_reason"]

    def test_a_fully_complete_coa_set_reports_nothing(self, client: TestClient) -> None:
        service = FakeCoaService([_complete_record()])
        app.dependency_overrides[get_coa_service] = lambda: service
        try:
            response = client.get("/missing-coas")
        finally:
            app.dependency_overrides.clear()

        assert response.status_code == 200
        assert response.json()["items"] == []

    def test_maps_a_mule_failure_to_503(
        self, client: TestClient, coa_service: FakeCoaService
    ) -> None:
        coa_service.error = MuleApiError(code="mule_down", message="mule is unreachable")

        response = client.get("/missing-coas")

        assert response.status_code == 503
        assert response.json()["detail"]["code"] == "mule_down"


class TestAdminUsers:
    def test_creates_an_external_account(self, client: TestClient) -> None:
        response = client.post(
            "/admin/users",
            json={
                "username": "dana.ops",
                "email": "dana.ops@example.com",
                "role": "3pl",
                "organization": "Harbor Freight",
            },
        )

        assert response.status_code == 201
        body = response.json()
        assert body["username"] == "dana.ops"
        assert body["email"] == "dana.ops@example.com"
        assert body["role"] == "3pl"
        assert body["organization"] == "Harbor Freight"
        assert body["user_id"]
        assert body["created_at"]

    def test_a_duplicate_username_is_a_409(self, client: TestClient) -> None:
        payload = {
            "username": "dana.ops",
            "email": "dana.ops@example.com",
            "role": "customer",
            "organization": None,
        }
        assert client.post("/admin/users", json=payload).status_code == 201

        response = client.post("/admin/users", json=payload)

        assert response.status_code == 409
        assert response.json()["code"] == "duplicate_user"
        assert response.json()["username"] == "dana.ops"

    def test_an_unknown_role_is_a_422(self, client: TestClient) -> None:
        response = client.post(
            "/admin/users",
            json={"username": "root.user", "email": "root@example.com", "role": "superuser"},
        )

        assert response.status_code == 422
        assert response.json()["code"] == "invalid_role"

    def test_a_malformed_email_is_a_422(self, client: TestClient) -> None:
        response = client.post(
            "/admin/users",
            json={"username": "bad.email", "email": "nope", "role": "internal"},
        )

        assert response.status_code == 422
        assert response.json()["code"] == "invalid_user"

    def test_a_missing_username_is_a_422(self, client: TestClient) -> None:
        response = client.post("/admin/users", json={"username": "", "role": "internal"})

        assert response.status_code == 422

    def test_a_customer_may_not_create_accounts(self, client: TestClient) -> None:
        response = client.post(
            "/admin/users",
            headers={"X-Role": "customer"},
            json={"username": "dana.ops", "role": "customer"},
        )

        assert response.status_code == 403
        assert response.json()["code"] == "forbidden"

    def test_an_internal_operator_may_create_accounts(self, client: TestClient) -> None:
        response = client.post(
            "/admin/users",
            headers={"X-Role": "internal"},
            json={"username": "dana.ops", "role": "internal"},
        )

        assert response.status_code == 201


class TestAuthLogin:
    def test_a_signed_token_is_exchanged_for_a_session(self, client: TestClient) -> None:
        response = client.post("/auth/login", json={"token": _hs256_token(_login_claims())})

        assert response.status_code == 200
        body = response.json()
        assert body["user_id"] == "user-4711"
        assert body["username"] == "dana.ops"
        assert body["roles"] == ["ventura-internal"]
        assert body["plants"] == ["PLANT-A"]
        assert body["expires_at"]

    def test_a_roles_claim_is_accepted(self, client: TestClient) -> None:
        claims = _login_claims(roles=["customer"], **{"cognito:groups": []})

        response = client.post("/auth/login", json={"token": _hs256_token(claims)})

        assert response.status_code == 200
        assert response.json()["roles"] == ["customer"]

    def test_a_garbage_token_is_a_401(self, client: TestClient) -> None:
        response = client.post("/auth/login", json={"token": "nope"})

        assert response.status_code == 401
        assert response.json() == {"detail": "invalid or expired token"}

    def test_an_expired_token_is_a_401(self, client: TestClient) -> None:
        claims = _login_claims(exp=int(time.time()) - 3600)

        response = client.post("/auth/login", json={"token": _hs256_token(claims)})

        assert response.status_code == 401

    def test_a_tampered_signature_is_a_401(self, client: TestClient) -> None:
        token = _hs256_token(_login_claims(), secret=os.urandom(48).hex())

        response = client.post("/auth/login", json={"token": token})

        assert response.status_code == 401

    def test_an_unsigned_token_is_a_401(self, client: TestClient) -> None:
        claims = _login_claims()
        header = _b64(json.dumps({"alg": "none", "typ": "JWT"}).encode("utf-8"))
        payload = _b64(json.dumps(claims).encode("utf-8"))

        response = client.post("/auth/login", json={"token": f"{header}.{payload}."})

        assert response.status_code == 401

    def test_a_token_without_an_expiry_is_a_401(self, client: TestClient) -> None:
        claims = _login_claims()
        claims.pop("exp")

        response = client.post("/auth/login", json={"token": _hs256_token(claims)})

        assert response.status_code == 401

    def test_an_empty_body_is_a_422(self, client: TestClient) -> None:
        assert client.post("/auth/login", json={}).status_code == 422


class StubReauthMintingPrintService:
    """Print service double that records what the re-auth endpoint asked it for.

    ``PrintService.issue_reauth_token`` is the single owner of the FR8 credential
    format, so the endpoint must delegate to it rather than mint one itself; these
    tests pin that delegation and the arguments it receives.
    """

    def __init__(
        self,
        token: str = "reauth.token.value",
        error: Optional[BaseException] = None,
        delegate: Optional[PrintService] = None,
    ) -> None:
        self.token = token
        self.error = error
        self.delegate = delegate
        self.mint_calls: list[tuple[str, str]] = []

    def issue_reauth_token(self, user_id: str, coa_id: str) -> str:
        self.mint_calls.append((user_id, coa_id))
        if self.error is not None:
            raise self.error
        return self.token

    def handle_print(self, request: Any) -> Any:
        if self.delegate is None:
            raise AssertionError("handle_print needs a delegate PrintService")
        return self.delegate.handle_print(request)


class TestPrintReauth:
    """``POST /coas/{coa_id}/print/reauth`` mints the credential ``/print`` needs."""

    def _with_service(
        self, monkeypatch: pytest.MonkeyPatch, service: StubReauthMintingPrintService
    ) -> StubReauthMintingPrintService:
        monkeypatch.setattr(contract_api, "build_print_service", lambda: service)
        return service

    def test_it_issues_the_reauth_token_for_the_requested_user_and_coa(
        self, client: TestClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        minted_token = os.urandom(12).hex()
        service = self._with_service(
            monkeypatch, StubReauthMintingPrintService(token=minted_token)
        )

        response = client.post(
            "/coas/COA-1001/print/reauth", json={"user_id": "user-4711"}
        )

        assert response.status_code == 200
        body = response.json()
        assert body["coa_id"] == "COA-1001"
        assert body["user_id"] == "user-4711"
        assert body["reauth_token"] == minted_token
        assert body["expires_at"]
        assert service.mint_calls == [("user-4711", "COA-1001")]

    def test_the_minted_token_is_accepted_by_the_print_endpoint(
        self, client: TestClient, monkeypatch: pytest.MonkeyPatch, print_service: PrintService
    ) -> None:
        """End to end: what the re-auth endpoint returns is what /print verifies.

        The token is signed the way ``PrintService._verify_reauth`` expects, so a
        successful print here proves the endpoint hands the portal a usable
        credential rather than an opaque string.
        """
        token = _reauth_token("user-4711", "COA-1001")
        self._with_service(
            monkeypatch, StubReauthMintingPrintService(token=token, delegate=print_service)
        )

        minted = client.post("/coas/COA-1001/print/reauth", json={"user_id": "user-4711"})
        assert minted.status_code == 200

        response = client.post(
            "/coas/COA-1001/print",
            json={"user_id": "user-4711", "reauth_token": minted.json()["reauth_token"]},
        )

        assert response.status_code == 200
        assert response.json()["success"] is True

    def test_a_deployment_without_a_print_service_is_a_503(
        self, client: TestClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from src.documents.print.print_service import ReauthError

        def _unconfigured() -> PrintService:
            raise ReauthError("PRINT_REAUTH_SECRET is not configured")

        monkeypatch.setattr(contract_api, "build_print_service", _unconfigured)

        response = client.post("/coas/COA-1001/print/reauth", json={"user_id": "user-4711"})

        assert response.status_code == 503
        assert response.json()["code"] == "reauth_unavailable"

    def test_a_minting_failure_is_surfaced_not_swallowed(
        self, client: TestClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        self._with_service(
            monkeypatch,
            StubReauthMintingPrintService(error=RuntimeError("signing key unavailable")),
        )

        response = client.post("/coas/COA-1001/print/reauth", json={"user_id": "user-4711"})

        assert response.status_code == 500
        assert response.json()["code"] == "reauth_issue_failed"
        assert response.json()["coa_id"] == "COA-1001"

    def test_the_user_id_is_required(self, client: TestClient) -> None:
        assert client.post("/coas/COA-1001/print/reauth", json={}).status_code == 422

    def test_an_empty_user_id_is_a_422(self, client: TestClient) -> None:
        response = client.post("/coas/COA-1001/print/reauth", json={"user_id": ""})

        assert response.status_code == 422

    def test_the_real_print_service_mints_a_token_the_print_endpoint_accepts(
        self, client: TestClient
    ) -> None:
        """The end-to-end proof, with no double in the loop.

        ``client`` wires ``build_print_service`` to a real
        :class:`PrintService`, so this drives the reviewer's complaint to ground:
        the portal obtains a credential from ``/print/reauth`` and that exact
        credential is what ``/print`` verifies.
        """
        minted = client.post("/coas/COA-1001/print/reauth", json={"user_id": "user-4711"})

        assert minted.status_code == 200
        token = minted.json()["reauth_token"]

        response = client.post(
            "/coas/COA-1001/print",
            json={"user_id": "user-4711", "reauth_token": token},
        )

        assert response.status_code == 200
        assert response.json()["success"] is True
        assert response.json()["audit_event"]["action"] == "print_success"

    def test_a_token_minted_for_one_coa_does_not_unlock_another(
        self, client: TestClient
    ) -> None:
        minted = client.post("/coas/COA-1001/print/reauth", json={"user_id": "user-4711"})

        response = client.post(
            "/coas/COA-1001/print",
            json={"user_id": "user-4711", "reauth_token": minted.json()["reauth_token"]},
        )
        assert response.status_code == 200

        # The same credential presented by a different user is refused.
        replayed = client.post(
            "/coas/COA-1001/print",
            json={"user_id": "user-9999", "reauth_token": minted.json()["reauth_token"]},
        )

        assert replayed.status_code == 401
        assert replayed.json()["code"] == "print_reauth_failed"
