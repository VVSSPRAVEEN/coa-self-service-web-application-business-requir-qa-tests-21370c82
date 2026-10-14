"""FR17: retry with exponential backoff, and the dead-letter queue.

Drives the real :mod:`src.integration.retry` decorator, classifier and DLQ. Three
guarantees are pinned:

* a **transient** failure is retried and a later success is returned to the
  caller, not swallowed;
* an **exhausted** retry budget writes a durable DLQ record carrying the attempt
  count and the failure, and re-raises;
* an **unrecoverable** failure goes straight to the DLQ without burning retries.

Retry delays are zeroed so these tests assert attempt *counts*, and the DLQ is
given a ``tmp_path`` so nothing is written into the repository.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from src.integration.retry import DeadLetterQueue, RetryPolicy, redrive_dlq, retry_call
from src.integration.retry.decorator import RETRY_METRICS
from src.integration.retry.errors import TransientError, UnrecoverableError, classify_error

FAST = RetryPolicy(base_delay=0.0, jitter=0.0, max_delay=0.0)


class Flaky:
    """A double that fails ``failures`` times before returning a value.

    ``retry_call`` reads ``fn.__name__`` when it logs and when it derives a DLQ
    record id, so the thing handed to the decorator has to be a real function.
    :attr:`fn` is that function; :attr:`attempts` is the count it accumulated.
    """

    def __init__(
        self,
        failures: int,
        value: Any = "ok",
        error: type[Exception] = TransientError,
    ) -> None:
        self.failures = failures
        self.value = value
        self.error = error
        self.attempts = 0
        self.fn = self._build()

    def _build(self) -> Any:
        double = self

        def flaky_fn(*args: Any, **kwargs: Any) -> Any:
            return double(*args, **kwargs)

        return flaky_fn

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        self.attempts += 1
        if self.attempts <= self.failures:
            raise self.error("simulated failure")
        return self.value


@pytest.fixture
def dlq(tmp_path: Path) -> DeadLetterQueue:
    return DeadLetterQueue(storage_path=str(tmp_path / "dlq"))


class TestRetryCall:
    def test_a_first_attempt_success_returns_immediately(self) -> None:
        flaky = Flaky(0)

        assert retry_call(policy=FAST)(flaky.fn)() == "ok"
        assert flaky.attempts == 1

    def test_a_transient_failure_is_retried_until_it_succeeds(self) -> None:
        flaky = Flaky(2)

        assert retry_call(policy=FAST)(flaky.fn)() == "ok"
        assert flaky.attempts == 3

    def test_the_arguments_are_passed_through_on_every_attempt(self) -> None:
        seen: list[tuple[tuple[Any, ...], dict[str, Any]]] = []

        @retry_call(policy=FAST)
        def record(*args: Any, **kwargs: Any) -> str:
            seen.append((args, kwargs))
            return "ok"

        record("a", key="v")

        assert seen == [(("a",), {"key": "v"})]

    def test_an_exhausted_budget_re_raises_the_last_error(self) -> None:
        flaky = Flaky(99)

        with pytest.raises(TransientError):
            retry_call(policy=FAST)(flaky.fn)()

        assert flaky.attempts == FAST.max_attempts

    def test_the_retry_count_matches_the_policy(self) -> None:
        flaky = Flaky(99)
        policy = RetryPolicy(base_delay=0.0, jitter=0.0, max_delay=0.0, max_attempts=4)

        with pytest.raises(TransientError):
            retry_call(policy=policy)(flaky.fn)()

        assert flaky.attempts == 4

    def test_an_unrecoverable_failure_is_not_retried(self) -> None:
        flaky = Flaky(99, error=UnrecoverableError)

        with pytest.raises(UnrecoverableError):
            retry_call(policy=FAST)(flaky.fn)()

        assert flaky.attempts == 1

    def test_the_wrapper_keeps_the_wrapped_function_sensible(self) -> None:
        @retry_call(policy=FAST)
        def documented() -> None:
            """A docstring the wrapper must not swallow."""

        assert documented.__name__ == "documented"
        assert documented.__doc__ == "A docstring the wrapper must not swallow."


class TestRetryMetrics:
    def test_a_success_is_counted(self) -> None:
        before = RETRY_METRICS["retry_successes"]

        retry_call(policy=FAST)(Flaky(1).fn)()

        assert RETRY_METRICS["retry_successes"] == before + 1

    def test_an_exhaustion_is_counted(self) -> None:
        before = RETRY_METRICS["retry_exhaustions"]

        with pytest.raises(TransientError):
            retry_call(policy=FAST)(Flaky(99).fn)()

        assert RETRY_METRICS["retry_exhaustions"] == before + 1

    def test_an_unrecoverable_failure_is_counted_separately(self) -> None:
        before = RETRY_METRICS["retry_unrecoverable"]

        with pytest.raises(UnrecoverableError):
            retry_call(policy=FAST)(Flaky(99, error=UnrecoverableError).fn)()

        assert RETRY_METRICS["retry_unrecoverable"] == before + 1


class TestExhaustionReachesTheDlq:
    def test_an_exhausted_call_writes_a_dlq_record(self, dlq: DeadLetterQueue) -> None:
        with pytest.raises(TransientError):
            retry_call(policy=FAST, dlq=dlq, record_id_fn=lambda: "coa-1")(Flaky(99).fn)()

        records = dlq.get_all_records()
        assert len(records) == 1
        assert records[0].record_id == "coa-1"
        assert records[0].attempt_count == FAST.max_attempts
        assert records[0].error_type == "TransientError"

    def test_an_unrecoverable_call_writes_a_dlq_record_on_its_first_failure(
        self, dlq: DeadLetterQueue
    ) -> None:
        flaky = Flaky(99, error=UnrecoverableError)

        with pytest.raises(UnrecoverableError):
            retry_call(policy=FAST, dlq=dlq, record_id_fn=lambda: "coa-2")(flaky.fn)()

        records = dlq.get_all_records()
        assert len(records) == 1
        assert records[0].attempt_count == 1
        assert flaky.attempts == 1

    def test_a_recovered_call_writes_nothing(self, dlq: DeadLetterQueue) -> None:
        retry_call(policy=FAST, dlq=dlq, record_id_fn=lambda: "coa-3")(Flaky(2).fn)()

        assert dlq.get_all_records() == []

    def test_without_a_dlq_an_exhaustion_still_raises(self) -> None:
        with pytest.raises(TransientError):
            retry_call(policy=FAST)(Flaky(99).fn)()

    def test_a_stack_trace_is_recorded(self, dlq: DeadLetterQueue) -> None:
        with pytest.raises(TransientError):
            retry_call(policy=FAST, dlq=dlq, record_id_fn=lambda: "coa-4")(Flaky(99).fn)()

        record = dlq.get_all_records()[0]
        assert record.stack_trace.startswith("Traceback (most recent call last):")
        assert "TransientError: simulated failure" in record.stack_trace


class TestErrorClassification:
    @pytest.mark.parametrize(
        "error",
        [
            pytest.param(UnrecoverableError("bad"), id="unrecoverable_error"),
            pytest.param(ConnectionError("reset"), id="connection_error"),
            pytest.param(TimeoutError("slow"), id="timeout_error"),
            pytest.param(OSError("io"), id="os_error"),
            pytest.param(ValueError("unknown, so retried rather than dropped"), id="value_error"),
        ],
    )
    def test_errors_are_classified(self, error: Exception) -> None:
        assert classify_error(error) in {"transient", "unrecoverable"}

    def test_the_two_typed_errors_are_classified_exactly(self) -> None:
        assert classify_error(TransientError("retry me")) == "transient"
        assert classify_error(UnrecoverableError("do not retry")) == "unrecoverable"

    def test_a_5xx_status_is_transient(self) -> None:
        class HttpError(Exception):
            status_code = 503

        assert classify_error(HttpError()) == "transient"

    def test_a_429_is_transient(self) -> None:
        class HttpError(Exception):
            status_code = 429

        assert classify_error(HttpError()) == "transient"

    def test_a_4xx_status_is_unrecoverable(self) -> None:
        class HttpError(Exception):
            status_code = 404

        assert classify_error(HttpError()) == "unrecoverable"

    def test_an_unknown_error_is_retried_rather_than_dropped(self) -> None:
        class Surprising(Exception):
            status_code = 200

        assert classify_error(Surprising()) == "transient"


class TestRetryPolicy:
    def test_the_delay_doubles_between_attempts(self) -> None:
        policy = RetryPolicy(base_delay=1.0, multiplier=2.0, jitter=0.0, max_delay=60.0)

        assert [policy.get_delay(attempt) for attempt in range(4)] == [1.0, 2.0, 4.0, 8.0]

    def test_the_delay_is_capped(self) -> None:
        policy = RetryPolicy(base_delay=1.0, multiplier=10.0, jitter=0.0, max_delay=5.0)

        assert policy.get_delay(3) == 5.0

    def test_a_negative_attempt_is_rejected(self) -> None:
        with pytest.raises(ValueError):
            RetryPolicy().get_delay(-1)

    def test_should_retry_counts_the_first_attempt_as_used(self) -> None:
        policy = RetryPolicy(max_attempts=3)

        assert policy.should_retry(1) is True
        assert policy.should_retry(2) is True
        assert policy.should_retry(3) is False


class TestDeadLetterQueue:
    def test_a_record_is_persisted_to_disk(self, dlq: DeadLetterQueue) -> None:
        dlq.write("coa-1", {"coa_id": "COA-1"}, 3, TransientError("gave up"))

        store = dlq._storage_dir / "dead_letters.jsonl"
        payload = json.loads(store.read_text(encoding="utf-8").splitlines()[0])

        assert payload["record_id"] == "coa-1"
        assert payload["attempt_count"] == 3
        assert payload["error_type"] == "TransientError"

    def test_a_fresh_queue_reloads_what_was_written(self, dlq: DeadLetterQueue) -> None:
        dlq.write("coa-1", {"coa_id": "COA-1"}, 3, TransientError("gave up"))

        reopened = DeadLetterQueue(storage_path=str(dlq._storage_dir))

        assert [record.record_id for record in reopened.get_all_records()] == ["coa-1"]
        assert reopened.is_processed("coa-1") is True

    def test_a_written_record_counts_as_processed(self, dlq: DeadLetterQueue) -> None:
        dlq.write("coa-1", {}, 1, TransientError("gave up"))

        assert dlq.is_processed("coa-1") is True

    def test_records_round_trip_through_a_mapping(self, dlq: DeadLetterQueue) -> None:
        written = dlq.write("coa-1", {"coa_id": "COA-1"}, 2, TransientError("gave up"))

        assert json.loads(written.to_json())["record_id"] == written.record_id
        assert dlq.get_all_records()[0].payload == {"coa_id": "COA-1"}

    def test_the_write_counter_moves(self, dlq: DeadLetterQueue) -> None:
        before = dlq.get_metrics()["dlq_records_written"]

        dlq.write("coa-1", {}, 1, TransientError("gave up"))

        assert dlq.get_metrics()["dlq_records_written"] == before + 1


class TestRedrive:
    def test_unredriven_records_are_replayed(self, dlq: DeadLetterQueue) -> None:
        dlq.write("coa-1", {"id": 1}, 1, TransientError("gave up"))
        dlq.write("coa-2", {"id": 2}, 1, TransientError("gave up"))
        seen: list[Any] = []

        summary = redrive_dlq(dlq, seen.append)

        assert summary["succeeded"] == 2
        assert seen == [{"id": 1}, {"id": 2}]

    def test_a_redriven_record_is_not_replayed_twice(self, dlq: DeadLetterQueue) -> None:
        dlq.write("coa-1", {"id": 1}, 1, TransientError("gave up"))
        redrive_dlq(dlq, lambda _payload: None)
        seen: list[Any] = []

        summary = redrive_dlq(dlq, seen.append)

        assert summary["total"] == 0
        assert seen == []

    def test_a_handler_failure_is_counted_and_the_batch_continues(
        self, dlq: DeadLetterQueue
    ) -> None:
        dlq.write("coa-1", {"id": 1}, 1, TransientError("gave up"))
        dlq.write("coa-2", {"id": 2}, 1, TransientError("gave up"))

        def handler(payload: Any) -> None:
            if payload["id"] == 1:
                raise RuntimeError("still broken")

        summary = redrive_dlq(dlq, handler)

        assert summary["succeeded"] == 1
        assert summary["failed"] == 1

    def test_stop_on_error_halts_the_batch(self, dlq: DeadLetterQueue) -> None:
        dlq.write("coa-1", {"id": 1}, 1, TransientError("gave up"))
        dlq.write("coa-2", {"id": 2}, 1, TransientError("gave up"))

        def handler(payload: Any) -> None:
            raise RuntimeError("still broken")

        summary = redrive_dlq(dlq, handler, stop_on_error=True)

        assert summary["failed"] == 1
        assert summary["succeeded"] == 0

    def test_the_batch_size_bounds_one_invocation(self, dlq: DeadLetterQueue) -> None:
        for index in range(5):
            dlq.write(f"coa-{index}", {"id": index}, 1, TransientError("gave up"))
        seen: list[Any] = []

        summary = redrive_dlq(dlq, seen.append, batch_size=2)

        assert summary["total"] == 5
        assert summary["succeeded"] == 2
        assert len(seen) == 2
