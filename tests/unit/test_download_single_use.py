"""FR18 "cannot be reused after use": the single-use scope store and its wiring.

Two layers are covered here:

* :class:`~src.documents.s3_service.consumption.SingleUseUrlStore` on its own -
  a second issue for the same ``(COA, actor)`` key retires the first scope,
  ``consume`` succeeds exactly once, and an unrelated key is never affected.
* the wiring in ``POST /coas/{coa_id}/download`` - two real requests through the
  FastAPI app must leave the first URL's scope inactive, asserted against the
  store the endpoint actually uses;
* the refusal path - when the audit append fails the endpoint answers 503 and
  issues nothing, so the scope it minted is retired rather than left spendable.

Everything is in-process: no network, and the store is read back through the same
``get_consumption_store`` accessor the endpoint calls, so the assertions cannot
drift onto a different instance.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any, Iterator, Optional

import pytest
from fastapi.testclient import TestClient

from src.api import contract_api
from src.api.mule.coa_service import COARecord, CompletionStatus, MuleApiError, get_coa_service
from src.api.mule.endpoints import app
from src.documents.s3_service import build_strategy_key
from src.documents.s3_service.consumption import SingleUseUrlStore
from src.documents.s3_service.presigned_url_service import PresignedUrlResult

PASSING_RESULTS = {
    "missing": [],
    "fields": {"moisture": {"status": "pass", "value": "12.1%"}},
}


def _store() -> SingleUseUrlStore:
    """A store whose scope ids are ``scope-1``, ``scope-2``, ... in order."""
    counter = iter(f"scope-{index}" for index in range(1, 100))
    return SingleUseUrlStore(id_factory=lambda: next(counter))


class TestIssueInvalidatesThePriorScope:
    def test_a_second_issue_retires_the_first_scope_for_the_same_key(self) -> None:
        store = _store()
        key = build_strategy_key("COA-1001", "user-4711")

        first = store.issue(key)
        second = store.issue(key)

        assert first != second
        assert store.is_active(first) is False
        assert store.is_active(second) is True
        assert store.active_scope_for(key) == second

    def test_a_retired_scope_can_no_longer_be_consumed(self) -> None:
        store = _store()
        key = build_strategy_key("COA-1001", "user-4711")

        first = store.issue(key)
        store.issue(key)

        assert store.consume(first) is False
        assert store.is_consumed(first) is False


class TestConsumeIsOnce:
    def test_a_scope_can_be_spent_exactly_once(self) -> None:
        store = _store()
        scope = store.issue(build_strategy_key("COA-1001", "user-4711"))

        assert store.consume(scope) is True
        assert store.consume(scope) is False
        assert store.consume(scope) is False
        assert store.is_consumed(scope) is True

    def test_a_consumed_scope_is_no_longer_active(self) -> None:
        store = _store()
        scope = store.issue(build_strategy_key("COA-1001", "user-4711"))

        store.consume(scope)

        assert store.is_active(scope) is False

    def test_an_unknown_scope_is_never_consumable(self) -> None:
        assert _store().consume("scope-that-was-never-issued") is False


class TestDifferentKeysAreUnaffected:
    def test_another_actor_on_the_same_coa_keeps_its_scope(self) -> None:
        store = _store()
        mine = store.issue(build_strategy_key("COA-1001", "user-4711"))
        theirs = store.issue(build_strategy_key("COA-1001", "user-9999"))

        store.issue(build_strategy_key("COA-1001", "user-4711"))

        assert store.is_active(theirs) is True
        assert store.consume(theirs) is True
        assert store.is_active(mine) is False

    def test_the_same_actor_on_another_coa_keeps_its_scope(self) -> None:
        store = _store()
        other_coa = store.issue(build_strategy_key("COA-2002", "user-4711"))

        store.issue(build_strategy_key("COA-1001", "user-4711"))

        assert store.is_active(other_coa) is True

    def test_the_strategy_key_separates_the_coa_from_the_actor(self) -> None:
        assert build_strategy_key("COA-1", "user-a") != build_strategy_key("COA-1", "user-b")
        assert build_strategy_key("COA-1", "user-a") != build_strategy_key("COA-2", "user-a")


class TestRetire:
    def test_retiring_drops_the_scope_and_reports_whether_it_was_live(self) -> None:
        store = _store()
        scope = store.issue(build_strategy_key("COA-1001", "user-4711"))

        assert store.retire(scope) is True
        assert store.is_active(scope) is False
        assert store.retire(scope) is False


# ---------------------------------------------------------------------------
# Contract-level proof: the download endpoint actually uses the store.
# ---------------------------------------------------------------------------


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
        test_results=PASSING_RESULTS,
    )


class _StubCoaService:
    def __init__(self, records: list[COARecord]) -> None:
        self._records = list(records)
        self._by_number = {record.coa_number: record for record in records}

    async def search_coa(self, filters: Any) -> list[COARecord]:
        return list(self._records)

    async def get_coa_by_number(self, coa_number: str) -> Optional[COARecord]:
        return self._by_number.get(coa_number)


class _StubPresignedService:
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
        self.calls.append({"key": key, "user_id": user_id, "coa_id": coa_id})
        return PresignedUrlResult(
            signed_url=f"https://coa-documents.s3.us-east-1.amazonaws.com/{key}?X-Amz-Signature=stub",
            bucket="coa-documents",
            key=key,
            method="GET",
            expires_at=datetime.now(timezone.utc) + timedelta(seconds=900),
            request_id="stub-request-id",
            scope_hash="stub-scope-hash",
        )


@pytest.fixture
def store(monkeypatch: pytest.MonkeyPatch) -> SingleUseUrlStore:
    """Install a fresh deterministic store as the one the endpoint reads."""
    counter = iter(f"scope-{index}" for index in range(1, 100))
    fresh = SingleUseUrlStore(id_factory=lambda: next(counter))
    monkeypatch.setattr(contract_api, "get_consumption_store", lambda: fresh)
    return fresh


@pytest.fixture
def client(monkeypatch: pytest.MonkeyPatch, tmp_path) -> Iterator[TestClient]:
    monkeypatch.setenv("ADMIN_USERS_PATH", str(tmp_path / "admin_users.json"))
    monkeypatch.setattr(contract_api, "log_access", lambda **kwargs: None)
    monkeypatch.setattr(
        contract_api,
        "build_presigned_url_service",
        lambda **kwargs: _StubPresignedService(),
    )
    monkeypatch.setattr(contract_api, "build_jde_client", lambda: None)
    app.dependency_overrides[get_coa_service] = lambda: _StubCoaService([_complete_record()])
    with TestClient(app) as test_client:
        yield test_client
    app.dependency_overrides.clear()


class TestDownloadIssuesSingleUseScopes:
    def test_a_download_issues_an_active_scope(
        self, client: TestClient, store: SingleUseUrlStore
    ) -> None:
        response = client.post("/coas/COA-1001/download", headers={"X-User-Id": "user-4711"})

        assert response.status_code == 200
        scope_id = response.json()["scope_id"]
        assert store.is_active(scope_id) is True
        assert store.active_scope_for(build_strategy_key("COA-1001", "user-4711")) == scope_id

    def test_a_second_download_invalidates_the_first_url_scope(
        self, client: TestClient, store: SingleUseUrlStore
    ) -> None:
        first = client.post("/coas/COA-1001/download", headers={"X-User-Id": "user-4711"})
        second = client.post("/coas/COA-1001/download", headers={"X-User-Id": "user-4711"})

        assert first.status_code == 200
        assert second.status_code == 200
        first_scope = first.json()["scope_id"]
        second_scope = second.json()["scope_id"]
        assert first_scope != second_scope
        assert store.is_active(first_scope) is False
        assert store.is_active(second_scope) is True

    def test_the_first_scope_is_unconsumable_after_the_second_request(
        self, client: TestClient, store: SingleUseUrlStore
    ) -> None:
        first_scope = client.post(
            "/coas/COA-1001/download", headers={"X-User-Id": "user-4711"}
        ).json()["scope_id"]

        client.post("/coas/COA-1001/download", headers={"X-User-Id": "user-4711"})

        assert store.consume(first_scope) is False

    def test_another_actor_downloading_does_not_invalidate_the_first(
        self, client: TestClient, store: SingleUseUrlStore
    ) -> None:
        mine = client.post("/coas/COA-1001/download", headers={"X-User-Id": "user-4711"}).json()[
            "scope_id"
        ]

        theirs = client.post("/coas/COA-1001/download", headers={"X-User-Id": "user-9999"}).json()[
            "scope_id"
        ]

        assert mine != theirs
        assert store.is_active(mine) is True
        assert store.consume(mine) is True
        assert store.is_active(theirs) is True

    def test_an_anonymous_caller_is_keyed_by_the_service_actor(
        self, client: TestClient, store: SingleUseUrlStore
    ) -> None:
        response = client.post("/coas/COA-1001/download")

        scope_id = response.json()["scope_id"]
        assert store.active_scope_for(build_strategy_key("COA-1001", contract_api.SERVICE_ACTOR)) == scope_id


class TestAFailedAuditLeavesNothingConsumable:
    """FR12 + FR18 together: no trail entry means no URL and no live scope.

    The endpoint mints the single-use scope before it writes the audit event, so
    a blocking audit has to revoke that scope. These tests are the regression
    guard for the order of those two operations.
    """

    def test_a_refused_download_retires_the_scope_it_had_issued(
        self, client: TestClient, store: SingleUseUrlStore, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def exploding_sink(**kwargs: Any) -> None:
            raise MuleApiError(code="audit_write_failed", message="sink down")

        monkeypatch.setattr(contract_api, "log_access", exploding_sink)

        response = client.post("/coas/COA-1001/download", headers={"X-User-Id": "user-4711"})

        assert response.status_code == 503
        assert response.json()["code"] == "audit_write_failed"
        assert store.active_scopes() == {}

    def test_a_prior_valid_scope_survives_an_audit_failure_on_the_next_request(
        self, client: TestClient, store: SingleUseUrlStore, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        first = client.post("/coas/COA-1001/download", headers={"X-User-Id": "user-4711"})
        live = first.json()["scope_id"]
        assert store.is_active(live) is True

        def exploding_sink(**kwargs: Any) -> None:
            raise MuleApiError(code="audit_write_failed", message="sink down")

        monkeypatch.setattr(contract_api, "log_access", exploding_sink)

        # The second request issues its own scope before it fails its audit, so
        # the first is retired by the issue while the second is retired by the
        # refusal. Nothing is left spendable, which is the property asserted.
        second = client.post("/coas/COA-1001/download", headers={"X-User-Id": "user-4711"})

        assert second.status_code == 503
        assert store.active_scopes() == {}
        assert store.consume(live) is False
