"""FR12: the immutable, hash-chained audit trail.

Drives the real :class:`~src.audit.trail.service.AuditTrailService` over an
in-memory repository. The property that matters is tamper *evidence*: appending
builds a chain in which every entry commits to its predecessor, so any edit to
any historical entry is detectable and localises to the tampered index.

Tampering is simulated the only way it can be in practice: by handing the service
a copy of the chain in which one stored entry has been altered and the chain
metadata left alone, exactly as an attacker editing a database row would.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Iterable

import pytest
from pydantic import ValidationError

from src.audit.trail.chain import AuditChain
from src.audit.trail.models import AuditEventType, AuditLogEntry
from src.audit.trail.repository import AuditRepository, InMemoryAuditStorage
from src.audit.trail.service import AuditTrailService

GENESIS = "GENESIS"


@pytest.fixture
def service() -> AuditTrailService:
    return AuditTrailService(repository=AuditRepository(InMemoryAuditStorage()))


def _append(service: AuditTrailService, count: int) -> AuditTrailService:
    for index in range(count):
        service.record_event(AuditEventType.ACCESS, f"user-{index}", f"coa:{index}")
    return service


def _replay(entries: Iterable[AuditLogEntry]) -> AuditTrailService:
    """Return a service whose storage holds exactly ``entries``, in order.

    Rebuilding the repository from a list is how a tampered chain would present
    itself to the verifier, and it keeps the tamper cases on the public API
    instead of reaching into the storage object.
    """
    storage = InMemoryAuditStorage()
    for entry in entries:
        storage.append(entry)
    return AuditTrailService(repository=AuditRepository(storage))


def _tamper(service: AuditTrailService, index: int, **changes: object) -> AuditTrailService:
    """Return a service holding ``service``'s chain with one entry altered."""
    entries = list(service.get_all_events())
    entries[index] = entries[index].model_copy(update=changes)
    return _replay(entries)


class TestAppend:
    def test_the_first_entry_is_linked_to_genesis(self, service: AuditTrailService) -> None:
        entry = service.record_event(
            event_type=AuditEventType.ACCESS,
            user_sub="user-4711",
            resource_id="coa:COA-1001",
        )

        assert entry.previous_hash == GENESIS
        assert entry.sequence_number == 0
        assert entry.entry_hash

    def test_each_entry_commits_to_its_predecessor(self, service: AuditTrailService) -> None:
        first = service.record_event(AuditEventType.ACCESS, "user-1", "coa:COA-1001")
        second = service.record_event(AuditEventType.ACCESS, "user-2", "coa:COA-1002")

        assert second.previous_hash == first.entry_hash
        assert second.sequence_number == 1

    def test_sequence_numbers_are_dense_and_monotonic(self, service: AuditTrailService) -> None:
        _append(service, 5)

        assert [entry.sequence_number for entry in service.get_all_events()] == [0, 1, 2, 3, 4]

    def test_every_field_survives_the_round_trip(self, service: AuditTrailService) -> None:
        service.record_event(
            event_type=AuditEventType.DOWNLOAD,
            user_sub="user-4711",
            resource_id="coa:COA-1001",
            details="signed the URL",
            ip_address="10.0.0.7",
        )

        stored = service.get_all_events()[0]
        assert stored.event_type is AuditEventType.DOWNLOAD
        assert stored.user_sub == "user-4711"
        assert stored.resource_id == "coa:COA-1001"
        assert stored.details == "signed the URL"
        assert stored.ip_address == "10.0.0.7"

    def test_entries_are_addressable_by_resource_and_by_type(
        self, service: AuditTrailService
    ) -> None:
        service.record_event(AuditEventType.ACCESS, "user-1", "coa:COA-1001")
        service.record_event(AuditEventType.PRINT, "user-1", "coa:COA-1001")
        service.record_event(AuditEventType.ACCESS, "user-2", "coa:COA-2002")

        assert len(service.get_events_for_resource("coa:COA-1001")) == 2
        assert len(service.get_events_by_type(AuditEventType.ACCESS)) == 2

    def test_an_idempotency_key_does_not_create_a_duplicate(
        self, service: AuditTrailService
    ) -> None:
        first = service.record_event(
            AuditEventType.ACCESS, "user-1", "coa:COA-1001", idempotency_key="evt-1"
        )
        second = service.record_event(
            AuditEventType.ACCESS, "user-1", "coa:COA-1001", idempotency_key="evt-1"
        )

        assert second is first
        assert len(service.get_all_events()) == 1


class TestImmutabilityOfTheSurface:
    def test_the_repository_exposes_no_update_or_delete_path(self) -> None:
        assert not hasattr(AuditRepository, "update")
        assert not hasattr(AuditRepository, "delete")
        assert not hasattr(InMemoryAuditStorage, "update")
        assert not hasattr(InMemoryAuditStorage, "delete")

    def test_a_stored_entry_cannot_be_edited_in_place(self, service: AuditTrailService) -> None:
        entry = service.record_event(AuditEventType.ACCESS, "user-1", "coa:COA-1001")

        with pytest.raises(ValidationError):
            entry.details = "rewritten"


class TestChainVerificationHappyPath:
    def test_an_untouched_chain_verifies(self, service: AuditTrailService) -> None:
        assert _append(service, 4).verify_chain_integrity() == (True, None)

    def test_an_empty_chain_verifies(self, service: AuditTrailService) -> None:
        assert service.verify_chain_integrity() == (True, None)

    def test_the_raw_chain_verifier_agrees_with_the_service(
        self, service: AuditTrailService
    ) -> None:
        _append(service, 3)

        assert AuditChain.verify_integrity(service.get_all_events()) == (True, None)

    def test_the_hash_is_reproducible_from_the_entry(self) -> None:
        entry = AuditLogEntry(
            event_type=AuditEventType.ACCESS,
            user_sub="user-1",
            resource_id="coa:COA-1001",
            timestamp=datetime(2026, 3, 31, 12, 0, tzinfo=timezone.utc),
        )
        sealed = AuditChain.seal_entry(entry, GENESIS, 0)

        assert AuditChain.compute_hash(sealed) == sealed.entry_hash


class TestTamperDetection:
    def test_editing_a_detail_breaks_the_hash_at_that_index(
        self, service: AuditTrailService
    ) -> None:
        _append(service, 3)

        tampered = _tamper(service, 1, details="quietly rewritten")

        assert tampered.verify_chain_integrity() == (False, 1)

    def test_changing_the_actor_is_detected(self, service: AuditTrailService) -> None:
        _append(service, 3)

        valid, failed_at = _tamper(service, 2, user_sub="someone-else").verify_chain_integrity()

        assert valid is False
        assert failed_at == 2

    def test_changing_the_resource_is_detected(self, service: AuditTrailService) -> None:
        _append(service, 2)

        valid, failed_at = _tamper(
            service, 0, resource_id="coa:SOMETHING-ELSE"
        ).verify_chain_integrity()

        assert valid is False
        assert failed_at == 0

    def test_backdating_an_entry_is_detected(self, service: AuditTrailService) -> None:
        _append(service, 2)
        original = service.get_all_events()[0]

        valid, failed_at = _tamper(
            service, 0, timestamp=original.timestamp.replace(year=2020)
        ).verify_chain_integrity()

        assert valid is False
        assert failed_at == 0

    def test_breaking_the_link_between_two_entries_is_detected(
        self, service: AuditTrailService
    ) -> None:
        _append(service, 3)

        assert _tamper(service, 2, previous_hash=GENESIS).verify_chain_integrity() == (False, 2)

    def test_removing_an_entry_from_the_middle_is_detected(
        self, service: AuditTrailService
    ) -> None:
        _append(service, 3)
        entries = list(service.get_all_events())
        del entries[1]

        valid, failed_at = _replay(entries).verify_chain_integrity()

        assert valid is False
        assert failed_at == 1

    def test_reordering_two_entries_is_detected(self, service: AuditTrailService) -> None:
        _append(service, 3)
        entries = list(service.get_all_events())
        entries[1], entries[2] = entries[2], entries[1]

        valid, failed_at = _replay(entries).verify_chain_integrity()

        assert valid is False
        assert failed_at == 1

    def test_appending_to_a_tampered_chain_does_not_repair_it(
        self, service: AuditTrailService
    ) -> None:
        _append(service, 3)
        tampered = _tamper(service, 1, details="quietly rewritten")
        tampered.record_event(AuditEventType.ACCESS, "user-9", "coa:9")

        assert tampered.verify_chain_integrity()[0] is False

    def test_a_snapshot_comparison_localises_the_new_index(
        self, service: AuditTrailService
    ) -> None:
        _append(service, 3)
        trusted = list(service.get_all_events())

        service.record_event(AuditEventType.ACCESS, "user-9", "coa:9")

        assert service.detect_tampering(trusted) == [3]

    def test_an_unchanged_snapshot_reports_nothing(self, service: AuditTrailService) -> None:
        service.record_event(AuditEventType.ACCESS, "user-1", "coa:1")

        assert service.detect_tampering(list(service.get_all_events())) == []

    def test_a_truncated_chain_is_reported_as_changed(self, service: AuditTrailService) -> None:
        _append(service, 3)
        trusted = list(service.get_all_events())
        entries = list(service.get_all_events())
        del entries[2]

        assert _replay(entries).detect_tampering(trusted) == [2]
