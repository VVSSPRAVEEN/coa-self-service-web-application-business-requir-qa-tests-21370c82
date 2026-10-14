"""FR4: the query-filter translation point behind ``GET /coa/search``.

``src.discovery.search_filters`` is the only place where raw query parameters
become the filter mapping the COA service layer receives, so its semantics are
pinned here directly against the module rather than through an HTTP round trip.

Three contracts are asserted:

* ``parse_filters`` emits canonical keys only (``q``, ``plant``, ``status``,
  ``start_date``, ``end_date``), accepts the documented aliases, and omits every
  value that carries no filter at all;
* ``normalize_status`` folds the upstream vocabulary onto the three canonical
  completion statuses, and ``normalize_date`` coerces the accepted date shapes
  to ISO ``YYYY-MM-DD``;
* unusable values are dropped and logged rather than raised, which is what
  keeps a malformed query parameter from turning a search into a 500. The
  module's typed error, :class:`COAFilterValidationError`, is a ``ValueError``
  so a caller that wraps this translation in its own validation can re-raise it
  and still be caught by an existing ``except ValueError`` handler.
"""

from __future__ import annotations

import logging
from datetime import date, datetime, timezone
from types import SimpleNamespace
from typing import Any, Dict, List

import pytest

from src.discovery.search_filters import (
    COAFilterValidationError,
    FILTER_FIELDS,
    STATUS_ALIASES,
    VALID_STATUSES,
    apply_filters,
    matches_filters,
    normalize_date,
    normalize_status,
    parse_filters,
)

LOGGER_NAME = "src.discovery.search_filters"


def _coa(**overrides: Any) -> SimpleNamespace:
    fields: Dict[str, Any] = {
        "coa_number": "COA-1001",
        "customer_name": "Harbor Freight",
        "plant_code": "PLANT-A",
        "status": "completed",
        "issue_date": "2026-03-01",
    }
    fields.update(overrides)
    return SimpleNamespace(**fields)


class TestCanonicalMapping:
    def test_a_fully_populated_query_maps_onto_the_canonical_keys(self) -> None:
        filters = parse_filters(
            q="COA-1001",
            plant="PLANT-A",
            status="completed",
            start_date="2026-01-01",
            end_date="2026-03-31",
        )

        assert filters == {
            "q": "COA-1001",
            "plant": "PLANT-A",
            "status": "completed",
            "start_date": "2026-01-01",
            "end_date": "2026-03-31",
        }

    def test_the_mapping_only_ever_contains_declared_filter_fields(self) -> None:
        filters = parse_filters(
            q="COA-1001",
            plant="PLANT-A",
            status="complete",
            start_date="2026-01-01",
            end_date="2026-03-31",
            search="ignored",
            plant_code="PLANT-B",
            completion_status="failed",
            date_from="2025-01-01",
            date_to="2025-12-31",
        )

        assert tuple(filters) == ("q", "plant", "status", "start_date", "end_date")
        assert set(filters) <= set(FILTER_FIELDS)

    def test_a_query_with_no_parameters_produces_an_empty_mapping(self) -> None:
        assert parse_filters() == {}

    def test_values_are_stripped_and_blank_values_are_omitted(self) -> None:
        filters = parse_filters(q="  COA-1001  ", plant="   ", status="")

        assert filters == {"q": "COA-1001"}

    def test_the_canonical_key_order_is_stable(self) -> None:
        filters = parse_filters(end_date="2026-03-31", q="COA-1001", plant="PLANT-A")

        assert list(filters) == ["q", "plant", "end_date"]


class TestAliases:
    def test_the_friendly_aliases_map_onto_the_canonical_names(self) -> None:
        filters = parse_filters(
            search="COA-1001",
            plant_code="PLANT-B",
            completion_status="complete",
            date_from="2026-01-01",
            date_to="2026-03-31",
        )

        assert filters == {
            "q": "COA-1001",
            "plant": "PLANT-B",
            "status": "completed",
            "start_date": "2026-01-01",
            "end_date": "2026-03-31",
        }

    def test_the_canonical_name_wins_when_both_forms_are_supplied(self) -> None:
        filters = parse_filters(
            q="COA-1001",
            search="COA-9999",
            plant="PLANT-A",
            plant_code="PLANT-B",
            status="failed",
            completion_status="pending",
            start_date="2026-02-01",
            date_from="2020-01-01",
            end_date="2026-02-28",
            date_to="2020-12-31",
        )

        assert filters == {
            "q": "COA-1001",
            "plant": "PLANT-A",
            "status": "failed",
            "start_date": "2026-02-01",
            "end_date": "2026-02-28",
        }

    def test_an_empty_canonical_value_suppresses_its_alias(self) -> None:
        assert parse_filters(q="", search="COA-1001") == {"q": "COA-1001"}
        assert parse_filters(status="", completion_status="complete") == {}


class TestStatusNormalization:
    @pytest.mark.parametrize(
        ("supplied", "expected"),
        [
            ("complete", "completed"),
            ("COMPLETED", "completed"),
            ("done", "completed"),
            ("incomplete", "pending"),
            ("pending", "pending"),
            ("open", "pending"),
            ("in progress", "pending"),
            ("in-progress", "pending"),
            ("in_progress", "pending"),
            ("failed", "failed"),
            ("fail", "failed"),
            ("error", "failed"),
        ],
    )
    def test_upstream_vocabulary_folds_onto_the_canonical_statuses(
        self, supplied: str, expected: str
    ) -> None:
        assert normalize_status(supplied) == expected

    def test_every_declared_alias_resolves_to_a_canonical_status(self) -> None:
        assert {normalize_status(alias) for alias in STATUS_ALIASES} == set(VALID_STATUSES)
        assert set(VALID_STATUSES) == {"pending", "completed", "failed"}

    def test_an_unrecognised_status_is_kept_in_lower_case(self) -> None:
        assert normalize_status("Quarantined") == "quarantined"

    def test_an_absent_or_blank_status_normalizes_to_none(self) -> None:
        assert normalize_status(None) is None
        assert normalize_status("   ") is None

    def test_parse_filters_normalizes_the_status_it_emits(self) -> None:
        assert parse_filters(status="In Progress")["status"] == "pending"
        assert parse_filters(completion_status="DONE")["status"] == "completed"


class TestDateNormalization:
    @pytest.mark.parametrize(
        ("supplied", "expected"),
        [
            ("2026-03-01", "2026-03-01"),
            ("  2026-03-01  ", "2026-03-01"),
            ("2026/03/01", "2026-03-01"),
            ("2026-03-01T10:30:00Z", "2026-03-01"),
            ("2026-03-01T10:30:00+02:00", "2026-03-01"),
            (date(2026, 3, 1), "2026-03-01"),
            (datetime(2026, 3, 1, 10, 30, tzinfo=timezone.utc), "2026-03-01"),
        ],
    )
    def test_the_accepted_shapes_all_normalize_to_iso_day(
        self, supplied: Any, expected: str
    ) -> None:
        assert normalize_date(supplied) == expected

    def test_an_absent_or_blank_date_normalizes_to_none(self) -> None:
        assert normalize_date(None) is None
        assert normalize_date("   ") is None

    def test_an_unparsable_date_is_dropped_and_logged(self, caplog: pytest.LogCaptureFixture) -> None:
        with caplog.at_level(logging.WARNING, logger=LOGGER_NAME):
            assert normalize_date("31-03-2026", field_name="start_date") is None

        assert any(
            record.levelno == logging.WARNING
            and "start_date" in record.getMessage()
            and "31-03-2026" in record.getMessage()
            for record in caplog.records
        )

    def test_dates_are_normalized_on_the_way_into_the_mapping(self) -> None:
        filters = parse_filters(
            start_date="2026/01/01", end_date=datetime(2026, 3, 31, 8, 0, tzinfo=timezone.utc)
        )

        assert filters == {"start_date": "2026-01-01", "end_date": "2026-03-31"}


class TestInvalidValues:
    def test_an_unparsable_date_is_omitted_without_raising(self) -> None:
        filters = parse_filters(q="COA-1001", start_date="yesterday", end_date="2026-03-31")

        assert filters == {"q": "COA-1001", "end_date": "2026-03-31"}

    def test_a_query_of_nothing_but_junk_yields_no_filters(self) -> None:
        assert parse_filters(q="   ", plant="  ", start_date="soon", end_date="") == {}

    def test_a_malformed_value_cannot_break_the_remaining_filters(self) -> None:
        filters = parse_filters(plant="PLANT-A", status="complete", start_date="not-a-date")

        assert filters == {"plant": "PLANT-A", "status": "completed"}

    def test_an_inverted_date_range_is_passed_through_unchanged(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        with caplog.at_level(logging.WARNING, logger=LOGGER_NAME):
            filters = parse_filters(start_date="2026-03-31", end_date="2026-01-01")

        assert filters == {"start_date": "2026-03-31", "end_date": "2026-01-01"}
        assert any(record.levelno == logging.WARNING for record in caplog.records)


class TestTypedErrorContract:
    def test_the_typed_validation_error_is_exported_for_callers(self) -> None:
        module = __import__("src.discovery.search_filters", fromlist=["__all__"])

        assert "COAFilterValidationError" in module.__all__

    def test_the_typed_validation_error_is_a_value_error(self) -> None:
        assert issubclass(COAFilterValidationError, ValueError)

        with pytest.raises(ValueError) as excinfo:
            raise COAFilterValidationError("status 'complete?' is not a completion status")

        assert isinstance(excinfo.value, COAFilterValidationError)
        assert str(excinfo.value) == "status 'complete?' is not a completion status"


class TestFilterApplication:
    def test_an_empty_filter_set_matches_every_record(self) -> None:
        records = [_coa(), _coa(coa_number="COA-1002")]

        assert apply_filters(records, {}) == records
        assert all(matches_filters(record, {}) for record in records)

    def test_the_search_term_matches_number_or_customer_case_insensitively(self) -> None:
        records = [_coa(), _coa(coa_number="COA-1002", customer_name="Other Co")]

        assert [record.coa_number for record in apply_filters(records, {"q": "coa-1001"})] == [
            "COA-1001"
        ]
        assert [record.coa_number for record in apply_filters(records, {"q": "OTHER"})] == [
            "COA-1002"
        ]

    def test_the_plant_filter_is_case_insensitive(self) -> None:
        records = [_coa(), _coa(coa_number="COA-1002", plant_code="PLANT-B")]

        assert [record.coa_number for record in apply_filters(records, {"plant": "plant-b"})] == [
            "COA-1002"
        ]

    def test_the_status_filter_normalizes_both_sides(self) -> None:
        records = [_coa(), _coa(coa_number="COA-1002", status="In Progress")]

        assert [record.coa_number for record in apply_filters(records, {"status": "pending"})] == [
            "COA-1002"
        ]

    def test_the_date_range_is_inclusive_on_both_ends(self) -> None:
        records = [
            _coa(coa_number="COA-1001", issue_date="2026-01-01"),
            _coa(coa_number="COA-1002", issue_date="2026-02-15"),
            _coa(coa_number="COA-1003", issue_date="2026-03-31"),
        ]

        matched = apply_filters(records, {"start_date": "2026-01-01", "end_date": "2026-03-31"})

        assert [record.coa_number for record in matched] == [
            "COA-1001",
            "COA-1002",
            "COA-1003",
        ]
        assert [
            record.coa_number
            for record in apply_filters(records, {"start_date": "2026-02-01", "end_date": "2026-03-30"})
        ] == ["COA-1002"]

    def test_the_criteria_are_combined_conjunctively(self) -> None:
        records: List[SimpleNamespace] = [
            _coa(coa_number="COA-1001"),
            _coa(coa_number="COA-1002", plant_code="PLANT-B"),
        ]

        matched = apply_filters(records, {"plant": "PLANT-A", "q": "COA-1001"})

        assert [record.coa_number for record in matched] == ["COA-1001"]