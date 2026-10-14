"""JDE connector, JDE/RDS reconciliation, and the JDE-aware completion gate (FR6, FR14).

These tests prove the three claims the JDE integration has to stand behind:

* :class:`~src.integration.jde.reader.JDEFieldReader` and
  :class:`~src.integration.jde.status.JDECOAStatus` really read *live* JDE
  documents - every assertion below runs against an injected
  :class:`~src.integration.jde.client.RecordingTransport`, never a socket, and
  the recorded requests are checked so a reader cannot quietly evaluate a
  snapshot instead of calling JDE.
* :func:`~src.integration.jde.reconciliation.reconcile` fails closed: a required
  field JDE does not report is missing, never assumed passing.
* The HTTP contract prefers the live verdict - a COA whose Mule-carried record
  looks complete but whose live JDE fields are outstanding is refused with the
  JDE-sourced ``missing_fields`` list.

The transport is injectable on purpose, so this module is hermetic: no network,
no environment, no clock dependence.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import pytest
from fastapi.testclient import TestClient

from src.api import contract_api
from src.api.mule.coa_service import COARecord, CompletionStatus, get_coa_service
from src.api.mule.endpoints import app
from src.integration.jde import (
    CoaStatusPath,
    JDEClient,
    JDECOAStatus,
    JDEConfigurationError,
    JDEError,
    JDEFieldReader,
    JDEPayloadError,
    JDETransportError,
    RDSMetadataRepository,
    RecordingTransport,
    TestFieldsPath,
    reconcile,
    validate_release,
)

EXPECTED_FIELDS = ["moisture", "protein", "fat"]

PASSING_FIELDS = {
    "moisture": {"status": "pass", "value": "12.1%"},
    "protein": {"status": "pass", "value": "41.0%"},
    "fat": {"status": "pass", "value": "3.4%"},
}

PARTIAL_FIELDS = {
    "moisture": {"status": "pass", "value": "12.1%"},
    "protein": {"status": "pass", "value": "41.0%"},
}


def _transport(
    *,
    fields: Any = None,
    status: Any = None,
    error: BaseException | None = None,
) -> RecordingTransport:
    """Build a recording transport preloaded with the two live JDE documents."""
    transport = RecordingTransport(
        {
            TestFieldsPath: fields if fields is not None else {"fields": PASSING_FIELDS},
            CoaStatusPath: (
                status
                if status is not None
                else {"coa_id": "COA-1001", "completion_status": "completed"}
            ),
        }
    )
    if error is not None:
        transport.fail_with(error)
    return transport


class TestJDEFieldReader:
    def test_reads_the_live_test_fields_out_of_jde(self) -> None:
        transport = _transport()

        fields = JDEFieldReader(transport).read_test_fields("COA-1001")

        assert set(fields) == set(EXPECTED_FIELDS)
        assert fields["moisture"] == {"status": "pass", "value": "12.1%"}
        request = transport.last_request()
        assert request is not None
        assert request.path == TestFieldsPath
        assert request.params == {"field_id": "COA-1001"}

    def test_accepts_a_flat_field_document(self) -> None:
        transport = _transport(fields={"moisture": "pass", "protein": {"status": "fail"}})

        fields = JDEFieldReader(transport).read_test_fields("COA-1001")

        assert fields["moisture"] == {"status": "pass"}
        assert fields["protein"] == {"status": "fail"}

    def test_a_bare_measurement_counts_as_evidence_the_test_ran(self) -> None:
        transport = _transport(fields={"fields": {"moisture": "12.1%"}})

        fields = JDEFieldReader(transport).read_test_fields("COA-1001")

        assert fields["moisture"] == {"value": "12.1%"}

    def test_an_absent_field_is_normalised_to_an_empty_entry(self) -> None:
        transport = _transport(fields={"fields": {"moisture": None}})

        fields = JDEFieldReader(transport).read_test_fields("COA-1001")

        assert fields["moisture"] == {}

    def test_rejects_an_unreadable_payload(self) -> None:
        transport = _transport(fields=["moisture"])

        with pytest.raises(JDEPayloadError):
            JDEFieldReader(transport).read_test_fields("COA-1001")

    def test_requires_a_field_id(self) -> None:
        with pytest.raises(ValueError):
            JDEFieldReader(_transport()).read_test_fields("  ")

    def test_propagates_a_transport_failure(self) -> None:
        transport = _transport(error=JDETransportError("JDE is unreachable"))

        with pytest.raises(JDEError):
            JDEFieldReader(transport).read_test_fields("COA-1001")


class TestJDECOAStatus:
    def test_reads_the_live_lifecycle_status(self) -> None:
        transport = _transport(
            status={
                "coa_id": "COA-1001",
                "completion_status": "completed",
                "completed_at": "2026-03-01T10:00:00Z",
            }
        )

        status = JDECOAStatus(transport).fetch_coa_status("COA-1001")

        assert status["coa_id"] == "COA-1001"
        assert status["completion_status"] == "completed"
        assert status["is_complete"] is True
        assert status["completed_at"] == "2026-03-01T10:00:00Z"
        request = transport.last_request()
        assert request is not None
        assert request.path == CoaStatusPath
        assert request.params == {"coa_id": "COA-1001"}

    def test_a_pending_coa_is_not_final(self) -> None:
        status = JDECOAStatus(_transport(status={"completion_status": "pending"})).fetch_coa_status(
            "COA-1001"
        )

        assert status["completion_status"] == "pending"
        assert status["is_complete"] is False

    def test_a_document_with_no_status_is_not_proof_of_release(self) -> None:
        status = JDECOAStatus(_transport(status={"coa_id": "COA-1001"})).fetch_coa_status(
            "COA-1001"
        )

        assert status["completion_status"] is None
        assert status["is_complete"] is False

    def test_a_boolean_completed_flag_is_understood(self) -> None:
        status = JDECOAStatus(_transport(status={"completed": True})).fetch_coa_status("COA-1001")

        assert status["completion_status"] == "completed"

    def test_requires_a_coa_id(self) -> None:
        with pytest.raises(ValueError):
            JDECOAStatus(_transport()).fetch_coa_status("")


class TestRDSMetadataRepository:
    def test_reads_the_expected_fields_through_the_injected_provider(self) -> None:
        seen: list[str] = []

        def provider(catalog_id: str) -> list[str]:
            seen.append(catalog_id)
            return EXPECTED_FIELDS

        fields = RDSMetadataRepository(provider).get_expected_fields("grain-catalog")

        assert fields == EXPECTED_FIELDS
        assert seen == ["grain-catalog"]

    def test_accepts_a_provider_object(self) -> None:
        class Catalog:
            def get_expected_fields(self, catalog_id: str) -> list[str]:
                return ["moisture"] if catalog_id == "grain" else []

        assert RDSMetadataRepository(Catalog()).get_expected_fields("grain") == ["moisture"]

    def test_normalises_a_comma_separated_provider_answer(self) -> None:
        assert RDSMetadataRepository(lambda _id: " moisture , protein ,moisture").get_expected_fields(
            "c"
        ) == ["moisture", "protein"]

    def test_wraps_a_provider_failure(self) -> None:
        def provider(_catalog_id: str) -> list[str]:
            raise RuntimeError("the metadata database is unreachable")

        with pytest.raises(JDEError):
            RDSMetadataRepository(provider).get_expected_fields("grain")

    def test_requires_a_provider(self) -> None:
        with pytest.raises(ValueError):
            RDSMetadataRepository(None)


class TestReconcile:
    def test_a_fully_passing_coa_reconciles_as_complete(self) -> None:
        result = reconcile(PASSING_FIELDS, EXPECTED_FIELDS)

        assert result.complete is True
        assert result.present == EXPECTED_FIELDS
        assert result.missing == []

    def test_a_field_jde_does_not_report_is_missing_not_assumed(self) -> None:
        result = reconcile(PARTIAL_FIELDS, EXPECTED_FIELDS)

        assert result.complete is False
        assert result.present == ["moisture", "protein"]
        assert result.missing == ["fat"]

    def test_a_failed_field_is_missing(self) -> None:
        fields = dict(PASSING_FIELDS, protein={"status": "fail"})

        assert reconcile(fields, EXPECTED_FIELDS).missing == ["protein"]

    def test_extra_fields_jde_reports_are_noted_but_not_required(self) -> None:
        fields = dict(PASSING_FIELDS, ash={"status": "pass"})

        result = reconcile(fields, EXPECTED_FIELDS)

        assert result.complete is True
        assert result.extra == ["ash"]

    def test_an_empty_catalog_requires_nothing(self) -> None:
        assert reconcile({}, []).complete is True


class TestValidateRelease:
    def test_a_complete_coa_is_releasable(self) -> None:
        outcome = validate_release(
            {"coa_id": "COA-1001", "completion_status": "completed"},
            PASSING_FIELDS,
            EXPECTED_FIELDS,
        )

        assert outcome.coa_id == "COA-1001"
        assert outcome.completion_status == "complete"
        assert outcome.releasable is True
        assert outcome.missing_fields == []

    def test_missing_live_fields_block_the_release_with_their_names(self) -> None:
        outcome = validate_release(
            {"coa_id": "COA-1001", "completion_status": "completed"},
            PARTIAL_FIELDS,
            EXPECTED_FIELDS,
        )

        assert outcome.releasable is False
        assert outcome.completion_status == "incomplete"
        assert outcome.missing_fields == ["fat"]

    def test_an_unreleased_lifecycle_blocks_even_when_every_field_passes(self) -> None:
        outcome = validate_release(
            {"coa_id": "COA-1001", "completion_status": "pending"},
            PASSING_FIELDS,
            EXPECTED_FIELDS,
        )

        assert outcome.releasable is False
        assert outcome.missing_fields == []

    def test_the_outcome_has_the_same_shape_as_the_pure_gate(self) -> None:
        from src.validation.completion_gate import ValidationOutcome

        outcome = validate_release(
            {"coa_id": "COA-1001", "completion_status": "completed"},
            PASSING_FIELDS,
            EXPECTED_FIELDS,
        )

        assert isinstance(outcome, ValidationOutcome)
        assert set(outcome.to_dict()) == {
            "coa_id",
            "completion_status",
            "releasable",
            "missing_fields",
        }


class TestJDEClient:
    def test_release_outcome_reads_all_three_live_facts(self) -> None:
        transport = _transport(fields={"fields": PASSING_FIELDS})
        client = JDEClient(
            transport,
            RDSMetadataRepository(lambda _catalog_id: EXPECTED_FIELDS),
            catalog_id="grain",
        )

        outcome = client.release_outcome("COA-1001")

        assert outcome.releasable is True
        assert [request.path for request in transport.requests] == [
            CoaStatusPath,
            TestFieldsPath,
        ]

    def test_from_env_raises_when_jde_is_not_configured(self, monkeypatch) -> None:
        monkeypatch.delenv("JDE_BASE_URL", raising=False)

        with pytest.raises(JDEConfigurationError):
            JDEClient.from_env()


# ---------------------------------------------------------------------------
# Contract-level proof: the gate prefers live JDE state over the record.
# ---------------------------------------------------------------------------


def _complete_record() -> COARecord:
    """A COA whose Mule-carried snapshot claims every test passed."""
    return COARecord(
        coa_number="COA-1001",
        customer_id="CUST-77",
        plant_code="PLANT-A",
        lot_number="LOT-11",
        product_code="PROD-9",
        issue_date="2026-03-01",
        completion_status=CompletionStatus.COMPLETED,
        status="completed",
        test_results={"missing": [], "fields": PASSING_FIELDS},
    )


class _StubCoaService:
    def __init__(self, records: list[COARecord]) -> None:
        self._records = list(records)
        self._by_number = {record.coa_number: record for record in records}

    async def search_coa(self, filters: Any) -> list[COARecord]:
        return list(self._records)

    async def get_coa_by_number(self, coa_number: str) -> COARecord | None:
        return self._by_number.get(coa_number)


@pytest.fixture
def contract_client(monkeypatch: pytest.MonkeyPatch, tmp_path) -> Iterator[TestClient]:
    monkeypatch.setenv("ADMIN_USERS_PATH", str(tmp_path / "admin_users.json"))
    monkeypatch.setattr(contract_api, "log_access", lambda **kwargs: None)
    monkeypatch.setattr(contract_api, "log_generation", lambda **kwargs: None)
    app.dependency_overrides[get_coa_service] = lambda: _StubCoaService([_complete_record()])
    with TestClient(app) as client:
        yield client
    app.dependency_overrides.clear()


def _use_jde(monkeypatch: pytest.MonkeyPatch, **kwargs: Any) -> RecordingTransport:
    """Point the contract at a JDE client backed by a recording transport."""
    transport = _transport(**kwargs)
    client = JDEClient(
        transport,
        RDSMetadataRepository(lambda _catalog_id: EXPECTED_FIELDS),
        catalog_id="grain",
    )
    monkeypatch.setattr(contract_api, "build_jde_client", lambda: client)
    return transport


class TestContractUsesLiveJDEState:
    def test_a_missing_jde_field_refuses_a_coa_the_record_calls_complete(
        self, contract_client: TestClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _use_jde(
            monkeypatch,
            fields={"fields": PARTIAL_FIELDS},
            status={"coa_id": "COA-1001", "completion_status": "completed"},
        )

        response = contract_client.post("/coas/COA-1001/download")

        assert response.status_code == 409
        body = response.json()
        assert body["code"] == "completion_gate_failed"
        assert body["coa_id"] == "COA-1001"
        assert body["missing_fields"] == ["fat"]
        assert set(body) == {"detail", "code", "coa_id", "missing_fields"}

    def test_validate_reports_the_same_jde_sourced_missing_fields(
        self, contract_client: TestClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _use_jde(monkeypatch, fields={"fields": PARTIAL_FIELDS})

        body = contract_client.post("/coas/COA-1001/validate").json()

        assert body["releasable"] is False
        assert body["missing_fields"] == ["fat"]

    def test_a_fully_passing_live_coa_is_allowed(
        self, contract_client: TestClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _use_jde(monkeypatch, fields={"fields": PASSING_FIELDS})

        body = contract_client.post("/coas/COA-1001/validate").json()

        assert body["releasable"] is True
        assert body["missing_fields"] == []

    def test_a_failed_live_read_degrades_to_the_snapshot_and_says_so(
        self, contract_client: TestClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _use_jde(monkeypatch, error=JDETransportError("JDE is unreachable"))

        response = contract_client.post("/coas/COA-1001/validate")

        assert response.status_code == 200
        body = response.json()
        # The snapshot says complete, and the response says why that is weaker.
        assert body["releasable"] is True
        assert body["jde_degraded"] is True
        assert body["gate_source"] == "mule"
        assert "JDE is unreachable" in body["jde_error"]

    def test_without_a_jde_client_no_degradation_is_reported(
        self, contract_client: TestClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(contract_api, "build_jde_client", lambda: None)

        body = contract_client.post("/coas/COA-1001/validate").json()

        assert "jde_degraded" not in body
        assert body["releasable"] is True


class TestBuildJDEClientFactory:
    def test_is_none_when_jde_base_url_is_unset(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("JDE_BASE_URL", raising=False)

        assert contract_api.build_jde_client() is None

    def test_builds_a_client_when_jde_base_url_is_set(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("JDE_BASE_URL", "https://jde.internal.example/api")
        monkeypatch.setenv("JDE_FIELD_CATALOG_ID", "grain")

        client = contract_api.build_jde_client()

        assert isinstance(client, JDEClient)
        assert client.catalog_id == "grain"
