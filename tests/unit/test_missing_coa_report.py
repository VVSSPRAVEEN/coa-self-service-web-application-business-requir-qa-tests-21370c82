"""FR10: the missing-COA report service and its two rendered bodies.

Drives the real :class:`~src.reports.missing_coa.service.MissingCoaReportService`
against fake discovery/validation repositories and a fixed clock, so "age_days"
is a deterministic function of the fixtures rather than of when the suite ran.

Three properties matter and are pinned here:

* **Determinism** - rows are ordered by age descending, then COA id ascending.
  Two COAs of the same age must not be able to swap places between runs.
* **Filtering** - a record that is not blocking anything is dropped, not
  rendered with an empty reason.
* **Honest empty output** - an empty repository renders an empty report rather
  than a header with zero rows.
"""

from __future__ import annotations

from datetime import date, timedelta
from typing import Any, Iterable

import pytest

from src.reports.missing_coa import MissingCoaReportService, render_csv, render_text
from src.reports.missing_coa.models import MissingCoaRow, ReportResult

TODAY = date(2026, 3, 31)


class FakeRepository:
    """A report repository backed by a fixed list of raw records."""

    def __init__(self, records: Iterable[dict[str, Any]] = ()) -> None:
        self._records = [dict(record) for record in records]
        self.calls = 0

    def _fetch(self) -> list[dict[str, Any]]:
        self.calls += 1
        return [dict(record) for record in self._records]

    def fetch_missing_coas(self) -> list[dict[str, Any]]:
        return self._fetch()

    def fetch_failed_validations(self) -> list[dict[str, Any]]:
        return self._fetch()


class FailingRepository:
    """A repository whose reads fail, standing in for an unavailable dependency."""

    def fetch_missing_coas(self) -> list[dict[str, Any]]:
        raise RuntimeError("the discovery store is unavailable")

    def fetch_failed_validations(self) -> list[dict[str, Any]]:
        raise RuntimeError("the validation store is unavailable")


def _record(
    coa_id: str,
    *,
    days_overdue: int = 0,
    priority: str | None = "standard",
    ship_date: str | None = None,
    status: str = "missing",
    validation_passed: bool | None = None,
) -> dict[str, Any]:
    return {
        "coa_id": coa_id,
        "vessel": "MV Ventura",
        "voyage": "007E",
        "counterparty": "Harbor Freight",
        "priority": priority,
        "required_ship_date": ship_date
        if ship_date is not None
        else (TODAY - timedelta(days=days_overdue)).isoformat(),
        "validation_passed": validation_passed,
        "status": status,
    }


def _service(discovery: Any = None, validation: Any = None) -> MissingCoaReportService:
    return MissingCoaReportService(
        discovery_repo=discovery or FakeRepository(),
        validation_repo=validation or FakeRepository(),
        clock=lambda: TODAY,
    )


class TestRowSelection:
    def test_a_blocked_coa_becomes_a_row(self) -> None:
        report = _service(discovery=FakeRepository([_record("COA-1", days_overdue=3)])).generate_report()

        assert len(report.rows) == 1
        row = report.rows[0]
        assert row.coa_id == "COA-1"
        assert row.vessel_voyage == "MV Ventura/007E"
        assert row.counterparty == "Harbor Freight"
        assert row.age_days == 3

    def test_a_coa_with_nothing_outstanding_is_dropped(self) -> None:
        clean = _record(
            "COA-CLEAN",
            days_overdue=0,
            priority="standard",
            status="issued",
            validation_passed=True,
        )

        report = _service(discovery=FakeRepository([clean])).generate_report()

        assert report.rows == ()

    def test_a_record_without_a_coa_id_is_dropped(self) -> None:
        nameless = _record("COA-1")
        nameless["coa_id"] = ""

        report = _service(discovery=FakeRepository([nameless])).generate_report()

        assert report.rows == ()

    def test_both_repositories_contribute_rows(self) -> None:
        report = _service(
            discovery=FakeRepository([_record("COA-DISCOVERY")]),
            validation=FakeRepository([_record("COA-VALIDATION", validation_passed=False)]),
        ).generate_report()

        assert {row.coa_id for row in report.rows} == {"COA-DISCOVERY", "COA-VALIDATION"}

    def test_an_unavailable_repository_yields_an_empty_report_not_an_error(self) -> None:
        report = _service(discovery=FailingRepository(), validation=FailingRepository()).generate_report()

        assert report.rows == ()


class TestBlockingReason:
    def test_an_overdue_coa_names_the_days(self) -> None:
        report = _service(discovery=FakeRepository([_record("COA-1", days_overdue=7)])).generate_report()

        assert "overdue by 7 days" in report.rows[0].blocking_reason

    def test_a_failed_validation_is_named(self) -> None:
        report = _service(
            validation=FakeRepository([_record("COA-1", validation_passed=False)])
        ).generate_report()

        assert "failed validation" in report.rows[0].blocking_reason

    def test_a_missing_priority_is_named(self) -> None:
        report = _service(discovery=FakeRepository([_record("COA-1", priority=None)])).generate_report()

        assert "missing priority" in report.rows[0].blocking_reason

    def test_a_missing_ship_date_is_named(self) -> None:
        record = _record("COA-1")
        record["required_ship_date"] = None

        report = _service(discovery=FakeRepository([record])).generate_report()

        assert "missing ship date" in report.rows[0].blocking_reason
        assert report.rows[0].age_days == 0

    def test_a_future_ship_date_never_reports_negative_age(self) -> None:
        record = _record("COA-1", ship_date=(TODAY + timedelta(days=5)).isoformat())

        report = _service(discovery=FakeRepository([record])).generate_report()

        assert report.rows[0].age_days == 0

    def test_an_unparsable_ship_date_is_treated_as_absent(self) -> None:
        record = _record("COA-1", ship_date="not-a-date")

        report = _service(discovery=FakeRepository([record])).generate_report()

        assert report.rows[0].required_ship_date is None


class TestOrdering:
    def test_rows_are_sorted_by_age_descending(self) -> None:
        discovery = FakeRepository(
            [
                _record("COA-YOUNG", days_overdue=1),
                _record("COA-OLD", days_overdue=30),
                _record("COA-MID", days_overdue=10),
            ]
        )

        report = _service(discovery=discovery).generate_report()

        assert [row.coa_id for row in report.rows] == ["COA-OLD", "COA-MID", "COA-YOUNG"]

    def test_rows_of_equal_age_are_broken_by_coa_id(self) -> None:
        discovery = FakeRepository(
            [_record("COA-C"), _record("COA-A"), _record("COA-B")]
        )

        report = _service(discovery=discovery).generate_report()

        assert [row.coa_id for row in report.rows] == ["COA-A", "COA-B", "COA-C"]

    def test_the_order_does_not_depend_on_repository_order(self) -> None:
        records = [_record(f"COA-{index:02d}", days_overdue=index) for index in range(1, 6)]

        first = _service(discovery=FakeRepository(records)).generate_report()
        second = _service(discovery=FakeRepository(list(reversed(records)))).generate_report()

        assert [row.coa_id for row in first.rows] == [row.coa_id for row in second.rows]


class TestRenderedOutput:
    def test_the_text_body_carries_every_column_header(self) -> None:
        report = _service(discovery=FakeRepository([_record("COA-1", days_overdue=4)])).generate_report()

        for header in ("COA ID", "Vessel/Voyage", "Counterparty", "Priority", "Age (Days)"):
            assert header in report.text_body
        assert "COA-1" in report.text_body
        assert "MV Ventura/007E" in report.text_body

    def test_the_csv_body_has_a_header_row_and_one_row_per_coa(self) -> None:
        discovery = FakeRepository([_record("COA-1", days_overdue=4), _record("COA-2")])

        report = _service(discovery=discovery).generate_report()
        lines = [line for line in report.csv_body.splitlines() if line]

        assert lines[0] == "coa_id,vessel_voyage,counterparty,priority,required_ship_date,age_days,blocking_reason"
        assert len(lines) == 3
        assert lines[1].startswith("COA-1,")
        assert "MV Ventura/007E" in lines[1]

    def test_the_renderers_agree_with_the_row_tuple(self) -> None:
        discovery = FakeRepository([_record("COA-1", days_overdue=4)])

        report = _service(discovery=discovery).generate_report()

        assert report.text_body == render_text(report.rows)
        assert report.csv_body == render_csv(report.rows)

    def test_a_row_without_a_voyage_renders_the_bare_vessel(self) -> None:
        record = _record("COA-1")
        record["voyage"] = ""

        report = _service(discovery=FakeRepository([record])).generate_report()

        assert report.rows[0].vessel_voyage == "MV Ventura"


class TestEmptyReport:
    def test_an_empty_repository_renders_an_empty_report(self) -> None:
        report = _service().generate_report()

        assert isinstance(report, ReportResult)
        assert report.rows == ()
        assert report.text_body.strip() == "No missing COA records found."
        assert report.csv_body.strip() == (
            "coa_id,vessel_voyage,counterparty,priority,required_ship_date,age_days,blocking_reason"
        )

    def test_an_empty_report_still_names_its_window_and_stamp(self) -> None:
        report = _service().generate_report(report_window="weekly")

        assert report.report_window == "weekly"
        assert report.generated_at

    def test_no_repositories_at_all_still_produce_a_report(self) -> None:
        report = MissingCoaReportService(clock=lambda: TODAY).generate_report()

        assert report.rows == ()


class TestInjectedRecords:
    def test_raw_records_bypass_the_repositories(self) -> None:
        discovery = FakeRepository([_record("COA-FROM-REPO")])

        report = _service(discovery=discovery).generate_report(
            raw_records=[_record("COA-INJECTED")], report_window="adhoc"
        )

        assert [row.coa_id for row in report.rows] == ["COA-INJECTED"]
        assert discovery.calls == 0
        assert report.report_window == "adhoc"


def test_the_row_model_is_frozen() -> None:
    row = MissingCoaRow(
        coa_id="COA-1",
        vessel="V",
        voyage="1",
        counterparty="C",
        priority="standard",
        required_ship_date=TODAY,
        age_days=0,
        blocking_reason="missing priority",
    )

    with pytest.raises(AttributeError):
        row.age_days = 5