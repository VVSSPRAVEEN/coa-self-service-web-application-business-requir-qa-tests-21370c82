"""FR16: near-real-time COA status synchronisation.

Drives the real :class:`~src.integration.sync.status_sync.StatusSyncService` and
its webhook adapter. The guarantee under test is that synchronisation is
*monotonic* and *idempotent*: a COA never regresses out of a terminal state, a
duplicate delta is absorbed rather than reapplied, and a stale out-of-order
message never overwrites newer state.

Timestamps are fixed so the suite never depends on the wall clock.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

import pytest

from src.integration.sync.models import (
    TERMINAL_STATUSES,
    VALID_TRANSITIONS,
    CoaStatus,
    StatusDelta,
    SyncResult,
)
from src.integration.sync.status_sync import StatusSyncService
from src.integration.sync.webhook import (
    WebhookValidationError,
    handle_batch_webhook,
    handle_status_webhook,
    validate_webhook_payload,
)

NOW = datetime(2026, 3, 31, 12, 0, 0)


def _delta(
    status: CoaStatus,
    *,
    sequence: int = 1,
    key: str = "evt-1",
    coa: str = "COA-1001",
    source: str = "JDE",
) -> StatusDelta:
    return StatusDelta(
        coa_number=coa,
        new_status=status,
        source=source,
        sequence_id=sequence,
        timestamp=NOW,
        idempotency_key=key,
    )


@pytest.fixture
def sync() -> StatusSyncService:
    return StatusSyncService()


class TestForwardProgress:
    def test_a_new_coa_starts_in_draft(self, sync: StatusSyncService) -> None:
        header = sync.get_or_create_header("COA-1001")

        assert header.current_status is CoaStatus.DRAFT
        assert header.last_sequence_id == 0

    def test_a_legal_transition_is_applied(self, sync: StatusSyncService) -> None:
        result = sync.apply_delta(_delta(CoaStatus.SUBMITTED))

        assert isinstance(result, SyncResult)
        assert result.applied is True
        assert result.previous_status is CoaStatus.DRAFT
        assert result.new_status is CoaStatus.SUBMITTED
        assert sync.get_header("COA-1001").current_status is CoaStatus.SUBMITTED

    def test_the_lifecycle_can_be_walked_forward_up_to_completed(
        self, sync: StatusSyncService
    ) -> None:
        path = [
            CoaStatus.SUBMITTED,
            CoaStatus.UNDER_REVIEW,
            CoaStatus.APPROVED,
            CoaStatus.COMPLETED,
        ]

        for index, status in enumerate(path, start=1):
            assert sync.apply_delta(_delta(status, sequence=index, key=f"evt-{index}")).applied

        assert sync.get_header("COA-1001").current_status is CoaStatus.COMPLETED

    def test_archived_is_unreachable_because_completed_is_already_terminal(
        self, sync: StatusSyncService
    ) -> None:
        """Pins a contradiction in the shipped state machine.

        ``VALID_TRANSITIONS`` declares ``COMPLETED -> ARCHIVED`` legal, but
        ``apply_delta`` consults its ``TERMINAL_STATUSES`` guard *before* the
        transition table, and COMPLETED is in that set. The declared edge is
        therefore unreachable and no COA can ever be archived. The test asserts
        the behaviour that actually ships, so the gap stays visible instead of
        being masked by a wish.
        """
        for index, status in enumerate(
            [CoaStatus.SUBMITTED, CoaStatus.UNDER_REVIEW, CoaStatus.APPROVED, CoaStatus.COMPLETED],
            start=1,
        ):
            sync.apply_delta(_delta(status, sequence=index, key=f"evt-{index}"))

        assert VALID_TRANSITIONS[CoaStatus.COMPLETED] == frozenset({CoaStatus.ARCHIVED})

        result = sync.apply_delta(_delta(CoaStatus.ARCHIVED, sequence=5, key="archive"))

        assert result.applied is False
        assert sync.get_header("COA-1001").current_status is CoaStatus.COMPLETED

    def test_a_rejection_is_a_legal_destination_from_draft(
        self, sync: StatusSyncService
    ) -> None:
        assert sync.apply_delta(_delta(CoaStatus.REJECTED)).applied is True
        assert sync.get_header("COA-1001").current_status is CoaStatus.REJECTED

    def test_the_sequence_id_advances_to_the_highest_seen(
        self, sync: StatusSyncService
    ) -> None:
        sync.apply_delta(_delta(CoaStatus.SUBMITTED, sequence=7, key="a"))
        sync.apply_delta(_delta(CoaStatus.UNDER_REVIEW, sequence=9, key="b"))

        assert sync.get_header("COA-1001").last_sequence_id == 9


class TestMonotonicity:
    def test_a_terminal_state_is_never_regressed(self, sync: StatusSyncService) -> None:
        sync.apply_delta(_delta(CoaStatus.REJECTED))

        result = sync.apply_delta(_delta(CoaStatus.SUBMITTED, sequence=2, key="later"))

        assert result.applied is False
        assert result.new_status is CoaStatus.REJECTED
        assert "terminal state" in result.reason

    def test_nothing_leaves_a_completed_coa(self, sync: StatusSyncService) -> None:
        for index, status in enumerate(
            [CoaStatus.SUBMITTED, CoaStatus.UNDER_REVIEW, CoaStatus.APPROVED, CoaStatus.COMPLETED],
            start=1,
        ):
            sync.apply_delta(_delta(status, sequence=index, key=f"evt-{index}"))

        for status in CoaStatus:
            assert sync.apply_delta(
                _delta(status, sequence=99, key=f"probe-{status.value}")
            ).applied is False

        assert sync.get_header("COA-1001").current_status is CoaStatus.COMPLETED

    def test_an_illegal_forward_jump_is_refused(self, sync: StatusSyncService) -> None:
        result = sync.apply_delta(_delta(CoaStatus.COMPLETED, sequence=1))

        assert result.applied is False
        assert "Illegal transition" in result.reason
        assert sync.get_header("COA-1001").current_status is CoaStatus.DRAFT

    @pytest.mark.parametrize("terminal", [CoaStatus.REJECTED, CoaStatus.COMPLETED])
    def test_a_reachable_terminal_status_is_absorbing(
        self, terminal: CoaStatus, sync: StatusSyncService
    ) -> None:
        """ARCHIVED is excluded: it is unreachable, see the test that says why."""
        sync.apply_delta(_delta(CoaStatus.SUBMITTED, key="a"))
        if terminal is CoaStatus.REJECTED:
            sync.apply_delta(_delta(CoaStatus.REJECTED, sequence=2, key="b"))
        else:
            for index, status in enumerate(
                [CoaStatus.UNDER_REVIEW, CoaStatus.APPROVED], start=2
            ):
                sync.apply_delta(_delta(status, sequence=index, key=f"s{index}"))
            sync.apply_delta(_delta(terminal, sequence=4, key="t"))

        for status in CoaStatus:
            sync.apply_delta(_delta(status, sequence=99, key=f"probe-{status.value}"))

        assert sync.get_header("COA-1001").current_status is terminal


class TestIdempotency:
    def test_the_same_delta_twice_is_applied_once(self, sync: StatusSyncService) -> None:
        first = sync.apply_delta(_delta(CoaStatus.SUBMITTED, key="evt-1"))
        second = sync.apply_delta(_delta(CoaStatus.SUBMITTED, sequence=5, key="evt-1"))

        assert first.applied is True
        assert second.applied is False
        assert "Idempotent duplicate" in second.reason
        assert sync.get_header("COA-1001").current_status is CoaStatus.SUBMITTED

    def test_a_duplicate_does_not_advance_the_sequence_id(
        self, sync: StatusSyncService
    ) -> None:
        sync.apply_delta(_delta(CoaStatus.SUBMITTED, sequence=3, key="evt-1"))

        sync.apply_delta(_delta(CoaStatus.SUBMITTED, sequence=99, key="evt-1"))

        assert sync.get_header("COA-1001").last_sequence_id == 3

    def test_distinct_keys_both_apply(self, sync: StatusSyncService) -> None:
        sync.apply_delta(_delta(CoaStatus.SUBMITTED, sequence=1, key="evt-1"))
        result = sync.apply_delta(_delta(CoaStatus.UNDER_REVIEW, sequence=2, key="evt-2"))

        assert result.applied is True


class TestOutOfOrder:
    def test_a_stale_delta_does_not_regress_newer_state(self, sync: StatusSyncService) -> None:
        sync.apply_delta(_delta(CoaStatus.SUBMITTED, sequence=5, key="new"))

        # APPROVED is not a legal destination from SUBMITTED, so this delta is
        # judged illegal *and* older than what has already been applied.
        result = sync.apply_delta(_delta(CoaStatus.APPROVED, sequence=2, key="old"))

        assert result.applied is False
        assert "Out-of-order stale delta" in result.reason
        assert sync.get_header("COA-1001").current_status is CoaStatus.SUBMITTED
        assert sync.get_header("COA-1001").last_sequence_id == 5

    def test_a_stale_delta_is_still_recorded_in_the_history(
        self, sync: StatusSyncService
    ) -> None:
        sync.apply_delta(_delta(CoaStatus.SUBMITTED, sequence=5, key="new"))
        sync.apply_delta(_delta(CoaStatus.REJECTED, sequence=2, key="old"))

        assert len(sync.get_sync_history("COA-1001")) == 2


class TestReconciliation:
    def test_an_externally_advanced_coa_is_caught_up(self, sync: StatusSyncService) -> None:
        sync.apply_delta(_delta(CoaStatus.SUBMITTED))

        results = sync.reconcile({"COA-1001": CoaStatus.UNDER_REVIEW}, "JDE")

        assert len(results) == 1
        assert results[0].applied is True
        assert sync.get_header("COA-1001").current_status is CoaStatus.UNDER_REVIEW

    def test_a_coa_already_in_step_produces_no_work(self, sync: StatusSyncService) -> None:
        sync.apply_delta(_delta(CoaStatus.SUBMITTED))

        assert sync.reconcile({"COA-1001": CoaStatus.SUBMITTED}, "JDE") == []

    def test_reconciliation_never_regresses_a_terminal_coa(self, sync: StatusSyncService) -> None:
        sync.apply_delta(_delta(CoaStatus.REJECTED))

        results = sync.reconcile({"COA-1001": CoaStatus.SUBMITTED}, "JDE")

        assert results == []
        assert sync.get_header("COA-1001").current_status is CoaStatus.REJECTED

    def test_reconciliation_refuses_an_illegal_jump(self, sync: StatusSyncService) -> None:
        assert sync.reconcile({"COA-1001": CoaStatus.ARCHIVED}, "JDE") == []
        assert sync.get_header("COA-1001").current_status is CoaStatus.DRAFT


class TestWebhookAdapter:
    def _payload(self, **overrides: Any) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "coa_number": "COA-1001",
            "status": "SUBMITTED",
            "source": "JDE",
            "sequence_id": 1,
            "timestamp": NOW.isoformat(),
        }
        payload.update(overrides)
        return payload

    def test_a_valid_payload_is_applied(self, sync: StatusSyncService) -> None:
        result = handle_status_webhook(self._payload(), sync)

        assert result.applied is True
        assert sync.get_header("COA-1001").current_status is CoaStatus.SUBMITTED

    def test_the_status_must_match_the_enum_spelling(self, sync: StatusSyncService) -> None:
        """``CoaStatus`` values are upper-case and the adapter does not fold case.

        Unlike ``source``, which is upper-cased before validation, ``status`` goes
        straight into ``CoaStatus(...)``, so a lower-case status is rejected
        rather than coerced. Pinned here so the asymmetry is a decision someone
        has to revisit, not an accident.
        """
        with pytest.raises(WebhookValidationError):
            handle_status_webhook(self._payload(status="submitted"), sync)

        # Validation runs before the sync service is touched, so no header exists.
        assert sync.get_header("COA-1001") is None

    def test_the_source_is_case_normalised(self, sync: StatusSyncService) -> None:
        handle_status_webhook(self._payload(source="jde"), sync)

        assert sync.get_header("COA-1001").current_status is CoaStatus.SUBMITTED

    def test_a_redelivered_webhook_is_absorbed(self, sync: StatusSyncService) -> None:
        handle_status_webhook(self._payload(), sync)
        result = handle_status_webhook(self._payload(sequence_id=42), sync)

        assert result.applied is False

    def test_a_payload_without_an_explicit_key_gets_a_deterministic_one(
        self, sync: StatusSyncService
    ) -> None:
        handle_status_webhook(self._payload(), sync)
        result = handle_status_webhook(self._payload(), sync)

        assert result.applied is False

    @pytest.mark.parametrize(
        "missing",
        ["coa_number", "status", "source", "sequence_id", "timestamp"],
    )
    def test_a_missing_required_field_is_rejected(self, missing: str, sync: StatusSyncService) -> None:
        payload = self._payload()
        del payload[missing]

        with pytest.raises(WebhookValidationError):
            handle_status_webhook(payload, sync)

    def test_an_unknown_source_is_rejected(self, sync: StatusSyncService) -> None:
        with pytest.raises(WebhookValidationError):
            handle_status_webhook(self._payload(source="SAP"), sync)

    def test_an_unknown_status_is_rejected(self, sync: StatusSyncService) -> None:
        with pytest.raises(WebhookValidationError):
            handle_status_webhook(self._payload(status="EXPLODED"), sync)

    def test_a_non_integer_sequence_id_is_rejected(self, sync: StatusSyncService) -> None:
        with pytest.raises(WebhookValidationError):
            handle_status_webhook(self._payload(sequence_id="1"), sync)

    def test_a_valid_payload_passes_validation_on_its_own(self) -> None:
        validate_webhook_payload(self._payload())

    def test_a_batch_skips_invalid_payloads_and_keeps_the_good_ones(
        self, sync: StatusSyncService
    ) -> None:
        results = handle_batch_webhook(
            [
                self._payload(status="SUBMITTED"),
                self._payload(status="EXPLODED", coa_number="COA-BAD"),
                self._payload(status="SUBMITTED", coa_number="COA-2002"),
            ],
            sync,
        )

        assert len(results) == 2
        assert sync.get_header("COA-1001").current_status is CoaStatus.SUBMITTED
        assert sync.get_header("COA-2002").current_status is CoaStatus.SUBMITTED


def test_the_transition_table_matches_the_terminal_guard() -> None:
    """The table and the guard must not disagree about where a COA can stop.

    ``ARCHIVED`` and ``REJECTED`` are absorbing in both. ``COMPLETED`` is the one
    place they diverge: the table allows ``ARCHIVED`` and the guard does not, so
    that edge is declared but unreachable.
    """
    absorbing = {CoaStatus.ARCHIVED, CoaStatus.REJECTED}

    for status in absorbing:
        assert VALID_TRANSITIONS[status] == frozenset(), status
        assert status in TERMINAL_STATUSES

    assert VALID_TRANSITIONS[CoaStatus.COMPLETED] == frozenset({CoaStatus.ARCHIVED})
    assert CoaStatus.COMPLETED in TERMINAL_STATUSES
