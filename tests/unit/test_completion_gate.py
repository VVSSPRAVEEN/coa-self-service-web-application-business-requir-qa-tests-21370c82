"""Unit tests for the COA completion gate and the release policy (FR6).

The gate is pure, so these tests need no service, no HTTP client and no
fixtures beyond the records themselves. They cover the two shapes a record
arrives in (``COARecord``-like object and plain mapping), the two ways the
required test fields are resolved (``test_results["missing"]`` and
``test_results["fields"]``), the fail-closed behaviour for a required field the
record never reports, and the typed error the HTTP layer maps to ``409``.
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from src.api.mule.coa_service import COARecord, CompletionStatus
from src.validation.completion_gate import (
    COMPLETE,
    INCOMPLETE,
    CompletionGate,
    CompletionGateError,
    GateDecision,
    ReleasePolicy,
    ValidationOutcome,
)

PASSING_RESULTS = {
    "missing": [],
    "fields": {
        "moisture": {"status": "pass", "value": "12.1%"},
        "protein": {"status": "pass", "value": "41.0%"},
    },
}

INCOMPLETE_RESULTS = {
    "missing": ["moisture", "protein"],
    "fields": {
        "moisture": {"status": "pending", "value": None},
        "protein": None,
    },
}


def _complete_record() -> COARecord:
    return COARecord(
        coa_number="COA-1001",
        plant_code="PLANT-A",
        completion_status=CompletionStatus.COMPLETED,
        status="completed",
        test_results=PASSING_RESULTS,
        generated_at=datetime(2026, 3, 1, 12, 0, tzinfo=UTC),
    )


def _incomplete_record() -> COARecord:
    return COARecord(
        coa_number="COA-1002",
        plant_code="PLANT-B",
        completion_status=CompletionStatus.PENDING,
        status="pending",
        test_results=INCOMPLETE_RESULTS,
    )


class TestCompletionGate:
    def test_complete_coa_is_releasable(self) -> None:
        outcome = CompletionGate().validate_coa(_complete_record())

        assert outcome.coa_id == "COA-1001"
        assert outcome.completion_status == COMPLETE
        assert outcome.releasable is True
        assert outcome.missing_fields == []

    def test_incomplete_coa_reports_the_missing_fields(self) -> None:
        outcome = CompletionGate().validate_coa(_incomplete_record())

        assert outcome.coa_id == "COA-1002"
        assert outcome.completion_status == INCOMPLETE
        assert outcome.releasable is False
        assert outcome.missing_fields == ["moisture", "protein"]

    def test_mapping_record_is_accepted(self) -> None:
        outcome = CompletionGate().validate_coa(
            {
                "coa_id": "COA-2001",
                "plant_of_origin": "PLANT-C",
                "completion_status": "completed",
                "test_results": PASSING_RESULTS,
            }
        )

        assert outcome.coa_id == "COA-2001"
        assert outcome.releasable is True

    def test_fields_without_a_missing_list_are_derived_from_the_field_results(self) -> None:
        outcome = CompletionGate().validate_coa(
            {
                "coa_id": "COA-2002",
                "status": "completed",
                "test_results": {
                    "fields": {
                        "moisture": {"status": "pass", "value": "12.1%"},
                        "protein": {"status": "fail", "value": None},
                        "ash": None,
                    }
                },
            }
        )

        assert outcome.missing_fields == ["protein", "ash"]
        assert outcome.releasable is False

    def test_plain_scalar_field_values_count_as_present(self) -> None:
        outcome = CompletionGate().validate_coa(
            {
                "coa_id": "COA-2003",
                "status": "completed",
                "test_results": {"fields": {"moisture": "12.1%", "protein": "41.0%"}},
            }
        )

        assert outcome.missing_fields == []
        assert outcome.releasable is True

    def test_explicit_required_fields_fail_closed_when_not_reported(self) -> None:
        gate = CompletionGate(required_fields=["moisture", "aflatoxin"])

        outcome = gate.validate_coa(
            {
                "coa_id": "COA-2004",
                "status": "completed",
                "test_results": {"fields": {"moisture": {"status": "pass", "value": "ok"}}},
            }
        )

        # aflatoxin is required and the record never reports it as passing.
        assert outcome.missing_fields == ["aflatoxin"]
        assert outcome.releasable is False

    def test_per_call_required_fields_override_the_configured_ones(self) -> None:
        gate = CompletionGate(required_fields=["aflatoxin"])

        outcome = gate.validate_coa(_complete_record(), required_fields=["moisture"])

        assert outcome.missing_fields == []
        assert outcome.releasable is True

    def test_pending_upstream_status_blocks_even_with_passing_tests(self) -> None:
        outcome = CompletionGate().validate_coa(
            {
                "coa_id": "COA-2005",
                "status": "completed",
                "completion_status": "pending",
                "test_results": PASSING_RESULTS,
            }
        )

        assert outcome.missing_fields == []
        assert outcome.completion_status == INCOMPLETE
        assert outcome.releasable is False

    def test_failed_upstream_status_blocks(self) -> None:
        outcome = CompletionGate().validate_coa(
            {"coa_id": "COA-2006", "status": "failed", "test_results": PASSING_RESULTS}
        )

        assert outcome.releasable is False

    def test_evaluation_is_deterministic(self) -> None:
        gate = CompletionGate()
        record = _incomplete_record()

        assert gate.validate_coa(record).to_dict() == gate.validate_coa(record).to_dict()

    def test_validate_coa_id_uses_the_loader(self) -> None:
        gate = CompletionGate()

        assert gate.validate_coa_id("COA-1001", lambda _id: _complete_record()).releasable is True
        assert gate.validate_coa_id("COA-1002", lambda _id: _incomplete_record()).releasable is False

    def test_validate_coa_id_reports_an_unknown_identifier(self) -> None:
        outcome = CompletionGate().validate_coa_id("COA-9999", lambda _id: None)

        assert outcome.coa_id == "COA-9999"
        assert outcome.releasable is False
        assert outcome.missing_fields == ["coa_record"]

    def test_missing_record_is_rejected(self) -> None:
        with pytest.raises(ValueError):
            CompletionGate().validate_coa(None)


class TestReleasePolicy:
    def test_allows_a_complete_coa(self) -> None:
        decision = ReleasePolicy().block_on_incomplete(_complete_record())

        assert isinstance(decision, GateDecision)
        assert decision.coa_id == "COA-1001"
        assert decision.allowed is True
        assert decision.missing_fields == []
        assert decision.reason == ReleasePolicy.REASON_RELEASED

    def test_blocks_an_incomplete_coa_with_its_missing_fields(self) -> None:
        decision = ReleasePolicy().block_on_incomplete(_incomplete_record())

        assert decision.allowed is False
        assert decision.missing_fields == ["moisture", "protein"]
        assert decision.reason == ReleasePolicy.REASON_INCOMPLETE

    def test_block_reason_mentions_the_upstream_status_when_tests_pass(self) -> None:
        decision = ReleasePolicy().block_on_incomplete(
            {
                "coa_id": "COA-3001",
                "status": "pending",
                "test_results": PASSING_RESULTS,
            }
        )

        assert decision.allowed is False
        assert decision.missing_fields == []
        assert decision.reason == ReleasePolicy.REASON_NOT_COMPLETED

    def test_require_releasable_returns_the_decision_when_allowed(self) -> None:
        decision = ReleasePolicy().require_releasable(_complete_record())

        assert decision.allowed is True

    def test_require_releasable_raises_with_the_missing_fields(self) -> None:
        with pytest.raises(CompletionGateError) as excinfo:
            ReleasePolicy().require_releasable(_incomplete_record())

        error = excinfo.value
        assert error.coa_id == "COA-1002"
        assert error.missing_fields == ["moisture", "protein"]
        assert error.to_dict()["code"] == "completion_gate_failed"
        assert error.to_dict()["coa_id"] == "COA-1002"

    def test_policy_accepts_an_injected_gate(self) -> None:
        policy = ReleasePolicy(CompletionGate(required_fields=["aflatoxin"]))

        decision = policy.block_on_incomplete(_complete_record())

        assert decision.allowed is False
        assert decision.missing_fields == ["aflatoxin"]


def test_validation_outcome_to_dict_is_json_ready() -> None:
    outcome = ValidationOutcome(
        coa_id="COA-4001",
        completion_status=INCOMPLETE,
        releasable=False,
        missing_fields=["moisture"],
    )

    assert outcome.to_dict() == {
        "coa_id": "COA-4001",
        "completion_status": "incomplete",
        "releasable": False,
        "missing_fields": ["moisture"],
    }
