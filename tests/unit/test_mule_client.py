"""FR4/FR12: the Mule COA client layer - request shape, parsing and auditing.

``COAService`` is the only component that talks to the Mule API, so the two
things a reviewer cannot see from the HTTP surface are pinned here against the
real service object: what is actually sent to Mule, and what the records that
come back are turned into.

The Mule client is replaced by a recorder double, which is the seam the service
was built for (``COAService.__init__`` takes the client as a dependency and the
audit trail is only touched through the module-level ``log_access``). The double
records every path it is handed and answers with a canned ``{"records": [...]}``
envelope; ``log_access`` is monkeypatched to a recorder for every test in this
module, so the suite proves the audit call happens - with the right actor,
action, resource and metadata - without opening an audit store.

``search_coa`` and ``get_coa_by_number`` deliberately differ: a search failure
propagates as the typed :class:`MuleApiError` so the route can answer 503, while
a detail lookup answers ``None`` so the route can answer 404. Both halves are
asserted.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

import pytest

from src.api.mule import coa_service as coa_service_module
from src.api.mule.coa_service import (
    ACCESS_READ,
    COAFilter,
    COARecord,
    COAService,
    CompletionStatus,
    MuleApiError,
)
from src.discovery.search_filters import parse_filters

AuditCall = tuple[tuple[str, str, str], dict[str, Any]]


@dataclass
class FakeResponse:
    """The object ``COAService`` receives back from ``ensure_ok``."""

    payload: dict[str, Any]

    def json(self) -> dict[str, Any]:
        return self.payload


class FakeMuleClient:
    """Records every call and answers from a canned payload.

    ``transport_error`` fails the call itself (the HTTP layer raised) and
    ``response_error`` fails ``ensure_ok`` (the request went out and came back
    as an error), which are the two failure shapes the service handles
    differently.
    """

    def __init__(
        self,
        payload: dict[str, Any] | None = None,
        *,
        transport_error: BaseException | None = None,
        response_error: BaseException | None = None,
    ) -> None:
        self.payload = payload if payload is not None else {"records": []}
        self.transport_error = transport_error
        self.response_error = response_error
        self.get_paths: list[str] = []
        self.post_calls: list[tuple[str, dict[str, Any]]] = []

    def get(self, path: str) -> FakeResponse:
        self.get_paths.append(path)
        if self.transport_error is not None:
            raise self.transport_error
        return FakeResponse(self.payload)

    def post(self, path: str, **kwargs: Any) -> FakeResponse:
        self.post_calls.append((path, kwargs))
        if self.transport_error is not None:
            raise self.transport_error
        return FakeResponse(self.payload)

    def ensure_ok(self, response: FakeResponse) -> FakeResponse:
        if self.response_error is not None:
            raise self.response_error
        return response


@pytest.fixture(autouse=True)
def audit_calls(monkeypatch: pytest.MonkeyPatch) -> list[AuditCall]:
    """Replace the module's audit sink with an in-process recorder.

    Applied to every test in this module: the audit trail is a persistence
    boundary of its own, already covered end to end by
    ``tests/unit/test_audit_chain.py``, and every read on this service is
    required to record an event. Tests that are not about the audit event itself
    assert against ``audit_calls`` rather than opening an audit store.
    """
    calls: list[AuditCall] = []

    def _record(
        actor: str,
        action: str,
        resource: str,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        calls.append(((actor, action, resource), dict(metadata or {})))

    monkeypatch.setattr(coa_service_module, "log_access", _record)
    return calls


def _mule_failure() -> MuleApiError:
    return MuleApiError(code="mule_down", message="the Mule API is unreachable")


def _search_payload() -> dict[str, Any]:
    return {
        "records": [
            {
                "coa_number": "COA-1001",
                "customer_id": "CUST-77",
                "plant_code": "PLANT-A",
                "lot_number": "LOT-11",
                "product_code": "PROD-9",
                "issue_date": "2026-03-01",
                "completion_status": "completed",
                "status": "completed",
                "file_size_bytes": 20481,
                "created_at": "2026-03-01T12:00:00",
            },
            {
                "coa_number": "COA-1002",
                "customer_id": "CUST-88",
                "plant_code": "PLANT-B",
                "completion_status": "failed",
                "created_at": "2026-03-01T07:00:00-05:00",
            },
            {
                "coa_number": "COA-1003",
                "plant_code": "PLANT-C",
                "completion_status": "awaiting-review",
            },
        ]
    }


class TestSearchRequestShape:
    async def test_every_non_empty_filter_reaches_the_mule_query_string(self) -> None:
        client = FakeMuleClient(_search_payload())

        await COAService(mule_client=client).search_coa(
            COAFilter(
                q="COA-1001",
                plant="PLANT-A",
                start_date="2026-01-01",
                end_date="2026-03-31",
                status="completed",
            )
        )

        assert client.get_paths == [
            "coa/search?q=COA-1001&plant=PLANT-A&start_date=2026-01-01&end_date=2026-03-31&status=completed"
        ]

    async def test_unset_filters_never_appear_in_the_query_string(self) -> None:
        client = FakeMuleClient(_search_payload())
        service = COAService(mule_client=client)

        await service.search_coa(COAFilter(plant="PLANT-B"))
        await service.search_coa(COAFilter())

        assert client.get_paths == ["coa/search?plant=PLANT-B", "coa/search"]

    async def test_only_the_filters_the_parser_kept_reach_the_request(self) -> None:
        client = FakeMuleClient(_search_payload())
        filters = COAFilter(**parse_filters(q="   ", plant="PLANT-A", status="complete"))

        await COAService(mule_client=client).search_coa(filters)

        assert client.get_paths == ["coa/search?plant=PLANT-A&status=completed"]


class TestSearchResponseParsing:
    async def test_the_record_envelope_becomes_coa_records(self) -> None:
        client = FakeMuleClient(_search_payload())

        records = await COAService(mule_client=client).search_coa(COAFilter())

        assert [record.coa_number for record in records] == ["COA-1001", "COA-1002", "COA-1003"]

        first = records[0]
        assert isinstance(first, COARecord)
        assert first.customer_id == "CUST-77"
        assert first.plant_code == "PLANT-A"
        assert first.lot_number == "LOT-11"
        assert first.product_code == "PROD-9"
        assert first.issue_date == "2026-03-01"
        assert first.file_size_bytes == 20481
        assert first.completion_status is CompletionStatus.COMPLETED
        assert first.status == "completed"

    async def test_the_completion_status_is_parsed_into_the_enum(self) -> None:
        client = FakeMuleClient(_search_payload())

        records = await COAService(mule_client=client).search_coa(COAFilter())

        assert records[1].completion_status is CompletionStatus.FAILED
        assert records[2].completion_status is None

    async def test_timestamps_are_read_as_aware_utc_instants(self) -> None:
        client = FakeMuleClient(_search_payload())

        records = await COAService(mule_client=client).search_coa(COAFilter())

        assert records[0].created_at == datetime(2026, 3, 1, 12, 0, tzinfo=UTC)
        assert records[1].created_at == datetime(2026, 3, 1, 12, 0, tzinfo=UTC)

    async def test_the_listing_response_does_not_invent_detail_only_fields(self) -> None:
        client = FakeMuleClient(_search_payload())

        records = await COAService(mule_client=client).search_coa(COAFilter())

        assert records[0].test_results is None
        assert records[0].generated_at is None
        assert records[0].completed_at is None

    async def test_an_empty_envelope_yields_no_records(self) -> None:
        client = FakeMuleClient({"records": []})

        assert await COAService(mule_client=client).search_coa(COAFilter()) == []

    async def test_a_search_failure_is_raised_to_the_caller(
        self, audit_calls: list[AuditCall]
    ) -> None:
        client = FakeMuleClient(_search_payload(), response_error=_mule_failure())

        with pytest.raises(MuleApiError) as excinfo:
            await COAService(mule_client=client).search_coa(COAFilter(q="COA-1001"))

        assert excinfo.value.code == "mule_down"
        assert audit_calls == []


class TestSearchAudit:
    async def test_the_search_is_written_to_the_audit_trail(
        self, audit_calls: list[AuditCall]
    ) -> None:
        client = FakeMuleClient(_search_payload())
        filters = COAFilter(q="COA-1001", plant="PLANT-A", status="completed")

        await COAService(mule_client=client).search_coa(filters)

        assert len(audit_calls) == 1
        identity, metadata = audit_calls[0]
        assert identity == ("coa_service", "search", "coa:search")
        assert metadata == {
            "filters": {
                "q": "COA-1001",
                "plant": "PLANT-A",
                "status": "completed",
            },
            "result_count": 3,
            "source": "mule_api",
        }

    async def test_the_recorded_event_type_is_the_access_read_vocabulary(self) -> None:
        assert ACCESS_READ == "coa_access_read"


class TestDetailLookup:
    def _detail_payload(self) -> dict[str, Any]:
        return {
            "customer_id": "CUST-77",
            "plant_code": "PLANT-A",
            "lot_number": "LOT-11",
            "product_code": "PROD-9",
            "issue_date": "2026-03-01",
            "completion_status": "completed",
            "status": "completed",
            "file_size_bytes": 20481,
            "test_results": {"missing": [], "fields": {}},
            "generated_at": "2026-03-01T12:00:00Z",
            "completed_at": "2026-03-01T12:05:00Z",
        }

    async def test_the_detail_path_is_the_coa_number(self) -> None:
        client = FakeMuleClient(self._detail_payload())

        await COAService(mule_client=client).get_coa_by_number("COA-1001")

        assert client.get_paths == ["coa/COA-1001"]

    async def test_a_slash_in_the_coa_number_cannot_escape_the_detail_path(self) -> None:
        client = FakeMuleClient(self._detail_payload())

        await COAService(mule_client=client).get_coa_by_number("COA/2026")

        assert client.get_paths == ["coa/COA%2F2026"]

    async def test_the_detail_payload_is_parsed_into_a_coa_record(self) -> None:
        client = FakeMuleClient(self._detail_payload())

        record = await COAService(mule_client=client).get_coa_by_number("COA-1001")

        assert record is not None
        assert record.coa_number == "COA-1001"
        assert record.customer_id == "CUST-77"
        assert record.plant_code == "PLANT-A"
        assert record.completion_status is CompletionStatus.COMPLETED
        assert record.test_results == {"missing": [], "fields": {}}
        assert record.generated_at == datetime(2026, 3, 1, 12, 0, tzinfo=UTC)
        assert record.completed_at == "2026-03-01T12:05:00Z"

    async def test_the_record_falls_back_to_the_requested_coa_number(self) -> None:
        client = FakeMuleClient(self._detail_payload())

        record = await COAService(mule_client=client).get_coa_by_number("COA-4242")

        assert record is not None
        assert record.coa_number == "COA-4242"

    @pytest.mark.parametrize("failure", ["transport", "response"])
    async def test_a_failed_lookup_answers_none_instead_of_raising(
        self, failure: str
    ) -> None:
        client = FakeMuleClient(
            self._detail_payload(),
            **(
                {"transport_error": ConnectionError("connection refused")}
                if failure == "transport"
                else {"response_error": _mule_failure()}
            ),
        )

        assert await COAService(mule_client=client).get_coa_by_number("COA-1001") is None

    async def test_a_malformed_timestamp_escapes_the_lookup(self) -> None:
        client = FakeMuleClient({"coa_number": "COA-1001", "generated_at": "not-a-timestamp"})

        with pytest.raises(ValueError):
            await COAService(mule_client=client).get_coa_by_number("COA-1001")

    async def test_the_detail_read_is_written_to_the_audit_trail(
        self, audit_calls: list[AuditCall]
    ) -> None:
        client = FakeMuleClient(self._detail_payload())

        await COAService(mule_client=client).get_coa_by_number("COA-1001")

        assert len(audit_calls) == 1
        identity, metadata = audit_calls[0]
        assert identity == ("coa_service", "read", "coa:COA-1001")
        assert metadata == {"coa_number": "COA-1001", "source": "mule_api"}

    async def test_a_failed_lookup_writes_no_audit_event(
        self, audit_calls: list[AuditCall]
    ) -> None:
        client = FakeMuleClient(self._detail_payload(), response_error=_mule_failure())

        assert await COAService(mule_client=client).get_coa_by_number("COA-1001") is None
        assert audit_calls == []