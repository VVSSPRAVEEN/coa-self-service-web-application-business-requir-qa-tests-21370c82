"""FR13: the five-year retention minimum and the legal-hold override.

Drives the real :mod:`src.audit.retention` policy, service and hold manager with
an injected reference time, so "is this record old enough to purge" is a
deterministic function of the fixtures rather than of when the suite runs.

**Naming note.** The requirement is often written as "a ``RetentionViolation`` on
early purge". The shipped policy expresses that guarantee as a *skip*, not an
exception: :meth:`RetentionPolicy.is_expired` returns False for a record inside
the window and :meth:`RetentionService.purge_expired_records` counts it under
``skipped_not_expired_count`` without ever calling the purge callback. These tests
assert that observable behaviour - a record inside the window is never destroyed
- rather than asserting an exception type the module does not define. The second
guarantee, a record under legal hold, is likewise a skip under
``skipped_held_count``.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from src.audit.retention import LegalHoldManager, RetentionPolicy, RetentionService
from src.audit.retention.policy import (
    DEFAULT_RETENTION_YEARS,
    MINIMUM_RETENTION_DAYS,
    RetentionConfig,
)

REFERENCE = datetime(2026, 3, 31, tzinfo=timezone.utc)
ISSUED = REFERENCE - timedelta(days=MINIMUM_RETENTION_DAYS)


class TestMinimumWindow:
    def test_the_minimum_is_five_years_in_days(self) -> None:
        assert MINIMUM_RETENTION_DAYS == 1826
        assert DEFAULT_RETENTION_YEARS == 5

    def test_the_default_policy_uses_the_minimum(self) -> None:
        policy = RetentionPolicy()

        assert policy.retention_days == MINIMUM_RETENTION_DAYS
        assert policy.validate_five_year_minimum() is True

    def test_the_expiry_date_is_exactly_1826_days_after_issuance(self) -> None:
        expiry = RetentionPolicy().get_expiry_date(ISSUED)

        assert expiry == ISSUED + timedelta(days=1826)
        assert (expiry - ISSUED).days == 1826

    def test_a_naive_timestamp_is_treated_as_utc(self) -> None:
        naive = datetime(2026, 3, 31)

        expiry = RetentionPolicy().get_expiry_date(naive)

        assert expiry == naive.replace(tzinfo=timezone.utc) + timedelta(days=1826)
        assert expiry.tzinfo is timezone.utc

    def test_a_configured_shorter_window_fails_the_five_year_check(self) -> None:
        policy = RetentionPolicy(RetentionConfig(retention_days=365))

        assert policy.validate_five_year_minimum() is False


class TestExpiryEvaluation:
    def test_a_record_inside_the_window_is_not_expired(self) -> None:
        policy = RetentionPolicy()

        assert policy.is_expired(REFERENCE - timedelta(days=MINIMUM_RETENTION_DAYS - 1), REFERENCE) is False

    def test_a_record_exactly_at_the_boundary_is_expired(self) -> None:
        assert RetentionPolicy().is_expired(ISSUED, REFERENCE) is True

    def test_a_record_past_the_window_is_expired(self) -> None:
        assert RetentionPolicy().is_expired(ISSUED - timedelta(days=1), REFERENCE) is True

    def test_the_remaining_days_count_down_to_the_expiry(self) -> None:
        policy = RetentionPolicy()

        assert policy.remaining_days(REFERENCE - timedelta(days=1000), REFERENCE) == 826

    def test_remaining_days_never_go_negative(self) -> None:
        assert RetentionPolicy().remaining_days(ISSUED - timedelta(days=10), REFERENCE) == 0


class TestEarlyPurgeIsRefused:
    def test_a_record_inside_the_window_is_never_destroyed(self) -> None:
        destroyed: list[str] = []

        result = RetentionService().purge_expired_records(
            [{"resource_id": "COA-1", "created_at": REFERENCE - timedelta(days=10)}],
            purge_callback=lambda resource_id: destroyed.append(resource_id) or True,
            reference_time=REFERENCE,
        )

        assert destroyed == []
        assert result.purged_count == 0
        assert result.skipped_not_expired_count == 1

    def test_a_record_past_the_window_is_destroyed(self) -> None:
        destroyed: list[str] = []

        result = RetentionService().purge_expired_records(
            [{"resource_id": "COA-1", "created_at": ISSUED}],
            purge_callback=lambda resource_id: destroyed.append(resource_id) or True,
            reference_time=REFERENCE,
        )

        assert destroyed == ["COA-1"]
        assert result.purged_count == 1
        assert result.purged_resource_ids == ["COA-1"]

    def test_an_iso_string_timestamp_is_accepted(self) -> None:
        destroyed: list[str] = []

        RetentionService().purge_expired_records(
            [{"resource_id": "COA-1", "created_at": (ISSUED).isoformat()}],
            purge_callback=lambda resource_id: destroyed.append(resource_id) or True,
            reference_time=REFERENCE,
        )

        assert destroyed == ["COA-1"]

    def test_a_malformed_timestamp_is_reported_not_purged(self) -> None:
        result = RetentionService().purge_expired_records(
            [{"resource_id": "COA-1", "created_at": "the day before yesterday"}],
            purge_callback=lambda _resource_id: True,
            reference_time=REFERENCE,
        )

        assert result.purged_count == 0
        assert result.errors

    def test_a_record_missing_its_identifier_is_reported_not_purged(self) -> None:
        result = RetentionService().purge_expired_records(
            [{"created_at": ISSUED}],
            purge_callback=lambda _resource_id: True,
            reference_time=REFERENCE,
        )

        assert result.purged_count == 0
        assert result.errors

    def test_a_refused_purge_callback_is_reported_as_an_error(self) -> None:
        result = RetentionService().purge_expired_records(
            [{"resource_id": "COA-1", "created_at": ISSUED}],
            purge_callback=lambda _resource_id: False,
            reference_time=REFERENCE,
        )

        assert result.purged_count == 0
        assert result.errors


class TestLegalHold:
    def test_a_held_record_is_never_destroyed_however_old_it_is(self) -> None:
        destroyed: list[str] = []
        service = RetentionService()
        service.place_legal_hold("COA-1", "litigation", "counsel@example.com")

        result = service.purge_expired_records(
            [{"resource_id": "COA-1", "created_at": ISSUED - timedelta(days=3650)}],
            purge_callback=lambda resource_id: destroyed.append(resource_id) or True,
            reference_time=REFERENCE,
        )

        assert destroyed == []
        assert result.purged_count == 0
        assert result.skipped_held_count == 1

    def test_releasing_the_hold_makes_the_record_purgeable(self) -> None:
        destroyed: list[str] = []
        service = RetentionService()
        service.place_legal_hold("COA-1", "litigation", "counsel@example.com")

        assert service.release_legal_hold("COA-1", "counsel@example.com") == 1

        service.purge_expired_records(
            [{"resource_id": "COA-1", "created_at": ISSUED}],
            purge_callback=lambda resource_id: destroyed.append(resource_id) or True,
            reference_time=REFERENCE,
        )

        assert destroyed == ["COA-1"]

    def test_a_hold_on_one_record_does_not_protect_another(self) -> None:
        destroyed: list[str] = []
        service = RetentionService()
        service.place_legal_hold("COA-1", "litigation", "counsel@example.com")

        service.purge_expired_records(
            [
                {"resource_id": "COA-1", "created_at": ISSUED},
                {"resource_id": "COA-2", "created_at": ISSUED},
            ],
            purge_callback=lambda resource_id: destroyed.append(resource_id) or True,
            reference_time=REFERENCE,
        )

        assert destroyed == ["COA-2"]

    def test_an_unknown_hold_reason_is_refused(self) -> None:
        service = RetentionService()

        with pytest.raises(ValueError):
            service.place_legal_hold("COA-1", "because-i-said-so", "counsel@example.com")

    def test_the_hold_manager_tracks_and_releases_holds(self) -> None:
        holds = LegalHoldManager()

        assert holds.is_on_hold("COA-1") is False
        holds.place_hold("COA-1", "regulatory_investigation", "auditor@example.com")
        assert holds.is_on_hold("COA-1") is True
        assert holds.get_all_held_resources() == {"COA-1"}
        assert holds.release_hold("COA-1", "auditor@example.com") == 1
        assert holds.is_on_hold("COA-1") is False

    def test_released_holds_remain_in_the_history(self) -> None:
        holds = LegalHoldManager()
        holds.place_hold("COA-1", "litigation", "counsel@example.com")
        holds.release_hold("COA-1", "counsel@example.com")

        history = holds.get_hold_history("COA-1")

        assert len(history) == 1
        assert history[0].is_active is False
        assert history[0].released_by == "counsel@example.com"

    def test_the_hold_manager_validates_reasons_when_asked_to(self) -> None:
        holds = LegalHoldManager()

        with pytest.raises(ValueError):
            holds.place_hold("COA-1", "vibes", "counsel@example.com", allowed_reasons={"litigation"})


class TestRecordEvaluation:
    def test_an_expired_unheld_record_is_eligible_for_purge(self) -> None:
        evaluation = RetentionService().evaluate_record("COA-1", ISSUED, REFERENCE)

        assert evaluation["is_expired"] is True
        assert evaluation["is_on_legal_hold"] is False
        assert evaluation["eligible_for_purge"] is True
        assert evaluation["remaining_days"] == 0

    def test_a_record_inside_the_window_is_not_eligible(self) -> None:
        evaluation = RetentionService().evaluate_record(
            "COA-1", REFERENCE - timedelta(days=10), REFERENCE
        )

        assert evaluation["is_expired"] is False
        assert evaluation["eligible_for_purge"] is False
        assert evaluation["remaining_days"] == 1816

    def test_a_held_record_is_never_eligible_however_old(self) -> None:
        service = RetentionService()
        service.place_legal_hold("COA-1", "customer_dispute", "support@example.com")

        evaluation = service.evaluate_record("COA-1", ISSUED - timedelta(days=3650), REFERENCE)

        assert evaluation["is_expired"] is True
        assert evaluation["eligible_for_purge"] is False

    def test_the_status_report_covers_every_usable_record(self) -> None:
        service = RetentionService()

        report = service.get_retention_status_report(
            [
                {"resource_id": "COA-1", "created_at": ISSUED},
                {"resource_id": "COA-2", "created_at": REFERENCE.isoformat()},
                {"resource_id": "COA-3", "created_at": "nonsense"},
                {"created_at": ISSUED},
            ],
            reference_time=REFERENCE,
        )

        assert [row["resource_id"] for row in report] == ["COA-1", "COA-2"]
