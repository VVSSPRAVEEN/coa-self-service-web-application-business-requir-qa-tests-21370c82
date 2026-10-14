"""The runtime job loop, driven with temporary directories and no network.

Every job in :mod:`src.workers.scheduler` is exercised here through
:meth:`Scheduler.run_once` - the same entry point the container loop calls - with
the clock, the HTTP fetcher and the mail transport all injected. Nothing in this
module opens a socket, and nothing writes outside the test's ``tmp_path``:

* ``bottomline_ingest`` gets real Bottomline files dropped into a temporary
  directory and asserts the existing ingestion pipeline parsed them and that the
  scheduler moved each one to ``./processed`` or ``./quarantine``.
* ``status_sync`` is asserted both ways: a no-op with no ``SYNC_BASE_URL`` (and
  proof the fetcher was never called), and a real reconciliation when one is
  configured and the injected fetcher returns a delta.
* ``dlq_redrive`` seeds the existing dead-letter queue with a replayable payload
  and asserts the pending record is redriven exactly once.
* ``notifications`` queues an alert on disk and asserts it goes out through the
  real :class:`EmailDispatcher` and the file leaves the queue.

The default transport is asserted to be the dry-run one, which is what keeps an
unconfigured worker from emitting mail.
"""

from __future__ import annotations

import base64
import json
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest

from src.integration.retry.dlq import DeadLetterQueue
from src.notifications.email.transport import DryRunTransport
from src.workers.scheduler import (
    BOTTOMLINE_INGEST,
    DLQ_REDRIVE,
    JOB_NAMES,
    NOTIFICATIONS,
    STATUS_SYNC,
    JobOutcome,
    Scheduler,
    SystemClock,
    main,
    run_once,
)

VALID_BOTTOMLINE = (
    "COA-1001|PROD-9|LOT-11|moisture|12.1|%|11.0|13.0|2026-03-01T00:00:00\n"
    "COA-1001|PROD-9|LOT-11|protein|41.0|%|40.0|42.0|2026-03-01T00:00:00"
)

MALFORMED_BOTTOMLINE = "this is not a bottomline flat file at all\n"


class FakeClock(SystemClock):
    """A clock that records sleeps instead of performing them."""

    def __init__(self) -> None:
        self.slept: list[float] = []

    def sleep(self, seconds: float) -> None:
        self.slept.append(seconds)


class RecordingFetcher:
    """A :class:`~src.workers.scheduler.Fetcher` that returns canned documents."""

    def __init__(self, documents: list[dict[str, Any]] | None = None) -> None:
        self.documents = list(documents or [])
        self.calls: list[str] = []

    def __call__(self, url: str) -> list[dict[str, Any]]:
        self.calls.append(url)
        return list(self.documents)


class InterruptingClock(FakeClock):
    """A clock whose first sleep ends the loop, so ``run_forever`` terminates."""

    def sleep(self, seconds: float) -> None:
        super().sleep(seconds)
        raise KeyboardInterrupt


@pytest.fixture
def workspace(tmp_path) -> dict[str, Path]:
    """A scheduler workspace entirely inside the test's temporary directory."""
    paths = {
        "root": tmp_path,
        "inbound": tmp_path / "inbound" / "bottomline",
        "processed": tmp_path / "processed",
        "quarantine": tmp_path / "quarantine",
        "alerts": tmp_path / "inbound" / "alerts",
        "dlq": tmp_path / "dlq_store",
    }
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    return paths


#: Workspace key -> (environment variable, path under the workspace root).
_DIRECTORIES = {
    "inbound": ("BOTTOMLINE_INBOUND_DIR", "inbound/bottomline"),
    "processed": ("BOTTOMLINE_PROCESSED_DIR", "processed"),
    "quarantine": ("BOTTOMLINE_QUARANTINE_DIR", "quarantine"),
    "alerts": ("ALERT_QUEUE_DIR", "inbound/alerts"),
    "dlq": ("DLQ_STORAGE_PATH", "dlq_store"),
}


def _env(workspace: Mapping[str, Path], **overrides: str) -> dict[str, str]:
    """Return an environment pointing every scheduler directory at ``workspace``.

    A partial mapping is accepted: any directory the caller did not name falls
    back to its default path under ``root``, so a test that exercises a single
    missing directory does not have to spell out the other four.
    """
    root = Path(workspace.get("root", Path.cwd()))
    env = {
        variable: str(workspace.get(key, root / relative))
        for key, (variable, relative) in _DIRECTORIES.items()
    }
    env["WORKER_INTERVAL_SECONDS"] = "5"
    env.update(overrides)
    return env


def _drop(directory: Path, name: str, content: str) -> Path:
    """Write one file into a drop directory and return its path."""
    path = directory / name
    path.write_text(content, encoding="utf-8")
    return path


def _outcome(outcomes: list[JobOutcome], name: str) -> JobOutcome:
    """Return the outcome recorded for one job."""
    return next(outcome for outcome in outcomes if outcome.name == name)


# ---------------------------------------------------------------------------
# (a) Bottomline ingestion
# ---------------------------------------------------------------------------


class TestBottomlineIngestJob:
    def test_a_well_formed_file_is_ingested_and_moved_to_processed(
        self, workspace
    ) -> None:
        _drop(workspace["inbound"], "BATCH_9001_001.txt", VALID_BOTTOMLINE)
        scheduler = Scheduler(env=_env(workspace))

        outcome = scheduler.run_bottomline_ingest()

        assert outcome.ok is True
        assert outcome.processed == 1
        assert list(workspace["inbound"].iterdir()) == []
        assert [p.name for p in workspace["processed"].iterdir()] == ["BATCH_9001_001.txt"]
        assert list(workspace["quarantine"].iterdir()) == []

    def test_the_existing_ingestion_pipeline_produced_the_records(
        self, workspace
    ) -> None:
        _drop(workspace["inbound"], "BATCH_9002_001.txt", VALID_BOTTOMLINE)
        scheduler = Scheduler(env=_env(workspace))

        scheduler.run_bottomline_ingest()

        assert len(scheduler._repository.get_records("9002")) == 2

    def test_an_unparsable_file_is_quarantined_not_retried_forever(
        self, workspace
    ) -> None:
        _drop(workspace["inbound"], "BATCH_9003_001.txt", MALFORMED_BOTTOMLINE)
        scheduler = Scheduler(env=_env(workspace))

        outcome = scheduler.run_bottomline_ingest()

        assert outcome.failed == 1
        assert outcome.processed == 0
        assert list(workspace["inbound"].iterdir()) == []
        quarantined = list(workspace["quarantine"].iterdir())
        assert len(quarantined) == 1
        assert quarantined[0].name.startswith("9003_")
        assert quarantined[0].name.endswith("_BATCH_9003_001.txt")
        assert quarantined[0].read_text(encoding="utf-8") == MALFORMED_BOTTOMLINE

    def test_a_replay_of_the_same_batch_is_recognised_as_a_duplicate(
        self, workspace
    ) -> None:
        scheduler = Scheduler(env=_env(workspace))
        _drop(workspace["inbound"], "BATCH_9004_001.txt", VALID_BOTTOMLINE)
        assert scheduler.run_bottomline_ingest().processed == 1

        _drop(workspace["inbound"], "BATCH_9004_001.txt", VALID_BOTTOMLINE)
        outcome = scheduler.run_bottomline_ingest()

        assert outcome.skipped == 1
        assert outcome.processed == 0
        assert [p.name for p in workspace["processed"].iterdir()] == [
            "BATCH_9004_001.txt",
            "BATCH_9004_001.1.txt",
        ]

    def test_a_missing_drop_directory_is_a_no_op_not_a_failure(self, tmp_path) -> None:
        outcome = Scheduler(env=_env({"root": tmp_path})).run_bottomline_ingest()

        assert outcome.ok is True
        assert outcome.skipped == 1
        assert "no drop directory" in outcome.detail

    def test_a_file_that_makes_the_ingestor_raise_is_dead_lettered_with_its_content(
        self, workspace, monkeypatch
    ) -> None:
        _drop(workspace["inbound"], "BATCH_9005_001.txt", VALID_BOTTOMLINE)

        def boom(_self: Any, _entry: Any) -> Any:
            raise RuntimeError("transformation exploded")

        monkeypatch.setattr(
            "src.integration.bottomline.ingestor.BottomlineIngestor.ingest_file", boom
        )
        scheduler = Scheduler(env=_env(workspace))

        outcome = scheduler.run_bottomline_ingest()

        assert outcome.failed == 1
        assert [p.name for p in workspace["quarantine"].iterdir()] == ["BATCH_9005_001.txt"]
        pending = DeadLetterQueue(str(workspace["dlq"])).get_unredriven_records()
        assert len(pending) == 1
        assert pending[0].error_type == "RuntimeError"
        assert base64.b64decode(pending[0].payload["content_b64"]).decode() == VALID_BOTTOMLINE


# ---------------------------------------------------------------------------
# (b) COA status sync
# ---------------------------------------------------------------------------


class TestStatusSyncJob:
    def test_it_is_a_no_op_when_no_base_url_is_configured(self, workspace) -> None:
        fetcher = RecordingFetcher()
        scheduler = Scheduler(env=_env(workspace), fetcher=fetcher)

        outcome = scheduler.run_status_sync()

        assert outcome.ok is True
        assert outcome.skipped == 1
        assert "SYNC_BASE_URL" in outcome.detail
        assert fetcher.calls == []

    def test_a_configured_endpoint_is_polled_and_applied_through_the_sync_api(
        self, workspace
    ) -> None:
        fetcher = RecordingFetcher(
            [
                {
                    "coa_number": "COA-1001",
                    "status": "SUBMITTED",
                    "source": "JDE",
                    "sequence_id": 7,
                    "timestamp": datetime.now(UTC).isoformat(),
                }
            ]
        )
        scheduler = Scheduler(
            env=_env(workspace, SYNC_BASE_URL="http://sync.internal:8081"),
            fetcher=fetcher,
        )

        outcome = scheduler.run_status_sync()

        assert fetcher.calls == ["http://sync.internal:8081/api/v1/sync/status"]
        assert outcome.processed == 1
        header = scheduler._sync_service.get_header("COA-1001")
        assert header is not None
        assert header.current_status.value == "SUBMITTED"

    def test_the_same_delta_twice_is_idempotent(self, workspace) -> None:
        delta = {
            "coa_number": "COA-1002",
            "status": "SUBMITTED",
            "source": "JDE",
            "sequence_id": 1,
            "timestamp": datetime.now(UTC).isoformat(),
        }
        scheduler = Scheduler(
            env=_env(workspace, SYNC_BASE_URL="http://sync.internal:8081"),
            fetcher=RecordingFetcher([delta]),
        )

        assert scheduler.run_status_sync().processed == 1
        assert scheduler.run_status_sync().processed == 0
        assert scheduler.run_status_sync().skipped == 1

    def test_an_invalid_payload_is_rejected_by_the_webhook_validator(
        self, workspace
    ) -> None:
        scheduler = Scheduler(
            env=_env(workspace, SYNC_BASE_URL="http://sync.internal:8081"),
            fetcher=RecordingFetcher([{"coa_number": "COA-1003"}]),
        )

        outcome = scheduler.run_status_sync()

        assert outcome.ok is True
        assert outcome.processed == 0
        assert scheduler._sync_service.get_header("COA-1003") is None


# ---------------------------------------------------------------------------
# (c) Dead-letter redrive
# ---------------------------------------------------------------------------


def _seed_dlq(workspace: Mapping[str, Path], *, replayable: bool) -> str:
    """Write one dead-letter record and return its id."""
    dlq = DeadLetterQueue(str(workspace["dlq"]))
    payload: dict[str, Any]
    if replayable:
        payload = {
            "batch_id": "9010",
            "file_name": "BATCH_9010_001.txt",
            "content_b64": base64.b64encode(VALID_BOTTOMLINE.encode("utf-8")).decode("ascii"),
        }
    else:
        payload = {"unrelated": "payload from another producer"}
    record = dlq.write(
        record_id="record-1",
        payload=payload,
        attempt_count=1,
        error=RuntimeError("earlier failure"),
        source="bottomline_ingest",
    )
    return record.record_id


class TestDlqRedriveJob:
    def test_a_pending_replayable_record_is_redriven_once(self, workspace) -> None:
        _seed_dlq(workspace, replayable=True)
        scheduler = Scheduler(env=_env(workspace))

        outcome = scheduler.run_dlq_redrive()

        assert outcome.ok is True
        assert outcome.processed == 1
        assert DeadLetterQueue(str(workspace["dlq"])).get_unredriven_records() == []
        assert len(scheduler._repository.get_records("9010")) == 2

    def test_a_redriven_record_is_not_replayed_twice(self, workspace) -> None:
        _seed_dlq(workspace, replayable=True)
        scheduler = Scheduler(env=_env(workspace))
        assert scheduler.run_dlq_redrive().processed == 1

        outcome = scheduler.run_dlq_redrive()

        assert outcome.processed == 0
        assert outcome.skipped == 0

    def test_a_payload_this_worker_cannot_interpret_stays_pending(self, workspace) -> None:
        _seed_dlq(workspace, replayable=False)
        scheduler = Scheduler(env=_env(workspace))

        outcome = scheduler.run_dlq_redrive()

        assert outcome.failed == 1
        assert outcome.processed == 0
        assert len(DeadLetterQueue(str(workspace["dlq"])).get_unredriven_records()) == 1

    def test_an_empty_queue_is_a_clean_no_op(self, workspace) -> None:
        outcome = Scheduler(env=_env(workspace)).run_dlq_redrive()
        assert outcome.ok is True
        assert outcome.processed == 0


# ---------------------------------------------------------------------------
# (d) Queued alert notifications
# ---------------------------------------------------------------------------


def _queue_alert(directory: Path, **overrides: Any) -> Path:
    """Write one alert document into the queue and return its path."""
    alert: dict[str, Any] = {
        "report_window": "2026-03-01",
        "subject": "Missing COA report 2026-03-01",
        "body_text": "One COA is blocking a shipment.",
        "csv_content": "coa_id,vessel\nCOA-1002,\n",
    }
    alert.update(overrides)
    path = directory / f"{alert['report_window']}.json"
    path.write_text(json.dumps(alert), encoding="utf-8")
    return path


class TestNotificationJob:
    def test_a_queued_alert_is_dispatched_and_removed_from_the_queue(
        self, workspace
    ) -> None:
        _queue_alert(workspace["alerts"])
        transport = DryRunTransport()
        scheduler = Scheduler(
            env=_env(workspace, ALERT_CS_RECIPIENTS="cs@example.com"),
            transport=transport,
        )

        outcome = scheduler.run_notifications()

        assert outcome.ok is True
        assert outcome.processed == 1
        assert list(workspace["alerts"].iterdir()) == []
        assert [p.name for p in workspace["processed"].iterdir()] == ["2026-03-01.json"]
        assert [message["recipients"] for message in transport.outbox] == [
            ["cs@example.com"]
        ]
        assert "Missing COA report" in transport.outbox[0]["subject"]

    def test_the_dispatched_message_is_rendered_not_merely_recorded(
        self, workspace
    ) -> None:
        _queue_alert(workspace["alerts"])
        transport = DryRunTransport()
        Scheduler(
            env=_env(workspace, ALERT_CS_RECIPIENTS="cs@example.com"),
            transport=transport,
        ).run_notifications()

        raw = transport.outbox[0]["raw_message"]
        assert "From: no-reply@ventura-coa.local" in raw
        assert "To: cs@example.com" in raw
        assert "missing-coa-report-2026-03-01.csv" in raw
        # Attachments travel base64-encoded, so decode to prove the payload.
        encoded = raw.split("Y29hX2lk")[1].split("\r\n")[0]
        assert base64.b64decode(f"Y29hX2lk{encoded}").decode() == (
            "coa_id,vessel\nCOA-1002,\n"
        )

    def test_an_unreadable_alert_is_quarantined(self, workspace) -> None:
        (workspace["alerts"] / "broken.json").write_text("{not json", encoding="utf-8")
        scheduler = Scheduler(env=_env(workspace, ALERT_CS_RECIPIENTS="cs@example.com"))

        outcome = scheduler.run_notifications()

        assert outcome.failed == 1
        assert [p.name for p in workspace["quarantine"].iterdir()] == ["broken.json"]
        assert list(workspace["alerts"].iterdir()) == []

    def test_an_alert_without_a_report_window_is_quarantined(self, workspace) -> None:
        (workspace["alerts"] / "incomplete.json").write_text(
            json.dumps({"subject": "no window"}), encoding="utf-8"
        )
        outcome = Scheduler(env=_env(workspace)).run_notifications()

        assert outcome.failed == 1
        assert [p.name for p in workspace["quarantine"].iterdir()] == ["incomplete.json"]

    def test_a_missing_queue_is_a_no_op(self, tmp_path) -> None:
        outcome = Scheduler(env=_env({"root": tmp_path})).run_notifications()

        assert outcome.ok is True
        assert outcome.skipped == 1
        assert "no alert queue" in outcome.detail

    def test_a_disallowed_channel_is_refused_and_quarantined(self, workspace) -> None:
        _queue_alert(workspace["alerts"], channel="slack")
        outcome = Scheduler(env=_env(workspace, ALERT_CS_RECIPIENTS="cs@example.com")).run_notifications()

        assert outcome.failed == 1
        assert [p.name for p in workspace["quarantine"].iterdir()] == ["2026-03-01.json"]


class TestTransportSelection:
    def test_an_unconfigured_worker_dispatches_into_a_dry_run_outbox(
        self, workspace
    ) -> None:
        _queue_alert(workspace["alerts"])
        scheduler = Scheduler(
            env=_env(workspace, ALERT_CS_RECIPIENTS="cs@example.com")
        )

        scheduler.run_notifications()

        assert isinstance(scheduler._transport, DryRunTransport)
        # dry_run stays False so the dispatcher really runs; the transport is
        # what keeps the message in-process.
        assert scheduler._dispatcher._settings.dry_run is False
        assert [message["recipients"] for message in scheduler._transport.outbox] == [
            ["cs@example.com"]
        ]

    def test_an_smtp_configuration_replaces_the_dry_run_transport(
        self, workspace
    ) -> None:
        from src.notifications.email.transport import SmtpMailTransport

        scheduler = Scheduler(env=_env(workspace, SMTP_HOST="smtp.example.com"))
        assert isinstance(scheduler._transport, SmtpMailTransport)

    def test_the_dry_run_flag_remains_available_as_an_operator_override(
        self, workspace
    ) -> None:
        scheduler = Scheduler(env=_env(workspace, ALERT_DRY_RUN="1"))
        assert scheduler._dispatcher._settings.dry_run is True


# ---------------------------------------------------------------------------
# The loop itself
# ---------------------------------------------------------------------------


class TestRunOnce:
    def test_run_once_executes_every_job_and_reports_each(self, workspace) -> None:
        _drop(workspace["inbound"], "BATCH_9100_001.txt", VALID_BOTTOMLINE)
        _queue_alert(workspace["alerts"])
        fetcher = RecordingFetcher()
        scheduler = Scheduler(
            env=_env(workspace, ALERT_CS_RECIPIENTS="cs@example.com"),
            clock=FakeClock(),
            fetcher=fetcher,
        )

        outcomes = scheduler.run_once()

        assert [outcome.name for outcome in outcomes] == list(JOB_NAMES)
        assert all(outcome.ok for outcome in outcomes)
        assert _outcome(outcomes, BOTTOMLINE_INGEST).processed == 1
        assert _outcome(outcomes, STATUS_SYNC).skipped == 1
        assert _outcome(outcomes, DLQ_REDRIVE).processed == 0
        assert _outcome(outcomes, NOTIFICATIONS).processed == 1
        assert fetcher.calls == []

    def test_run_once_moves_the_files_and_leaves_the_drop_directories_empty(
        self, workspace
    ) -> None:
        _drop(workspace["inbound"], "BATCH_9101_001.txt", VALID_BOTTOMLINE)
        _drop(workspace["inbound"], "BATCH_9102_001.txt", MALFORMED_BOTTOMLINE)
        _queue_alert(workspace["alerts"])
        scheduler = Scheduler(
            env=_env(workspace, ALERT_CS_RECIPIENTS="cs@example.com"),
            clock=FakeClock(),
        )

        scheduler.run_once()

        assert list(workspace["inbound"].iterdir()) == []
        assert sorted(p.name for p in workspace["processed"].iterdir()) == [
            "2026-03-01.json",
            "BATCH_9101_001.txt",
        ]
        quarantined = [p.name for p in workspace["quarantine"].iterdir()]
        assert len(quarantined) == 1
        assert quarantined[0].endswith("_BATCH_9102_001.txt")

    def test_one_failing_job_does_not_stop_the_others(self, workspace, monkeypatch) -> None:
        _queue_alert(workspace["alerts"])

        def boom(_self: Any) -> Any:
            raise RuntimeError("ingestor is unavailable")

        monkeypatch.setattr(
            "src.integration.bottomline.ingestor.BottomlineIngestor.poll_and_ingest", boom
        )
        monkeypatch.setattr(
            "src.integration.bottomline.sftp_poller.SftpPoller.poll",
            lambda self: (_ for _ in ()).throw(RuntimeError("poller is unavailable")),
        )
        scheduler = Scheduler(
            env=_env(workspace, ALERT_CS_RECIPIENTS="cs@example.com"),
            clock=FakeClock(),
        )

        outcomes = scheduler.run_once()

        failed = _outcome(outcomes, BOTTOMLINE_INGEST)
        assert failed.ok is False
        assert "poller is unavailable" in failed.detail
        assert _outcome(outcomes, NOTIFICATIONS).processed == 1

    def test_the_module_level_entry_point_runs_one_full_cycle(self, workspace) -> None:
        _drop(workspace["inbound"], "BATCH_9103_001.txt", VALID_BOTTOMLINE)

        outcomes = run_once(Scheduler(env=_env(workspace), clock=FakeClock()))

        assert [outcome.name for outcome in outcomes] == list(JOB_NAMES)
        assert [p.name for p in workspace["processed"].iterdir()] == ["BATCH_9103_001.txt"]

    def test_an_outcome_reports_its_own_counters(self) -> None:
        assert JobOutcome(name="x", processed=2, failed=1).as_log_fields() == {
            "job": "x",
            "job_ok": True,
            "job_processed": 2,
            "job_failed": 1,
            "job_skipped": 0,
        }


class TestRunForever:
    def test_the_loop_cycles_on_the_configured_interval(self, workspace) -> None:
        clock = FakeClock()

        class CountingScheduler(Scheduler):
            def __init__(self, **kwargs: Any) -> None:
                super().__init__(**kwargs)
                self.cycles = 0

            def run_once(self) -> list[JobOutcome]:
                self.cycles += 1
                if self.cycles == 3:
                    raise KeyboardInterrupt
                return super().run_once()

        scheduler = CountingScheduler(env=_env(workspace), clock=clock)
        scheduler._clock = clock

        assert scheduler.run_forever() == 0
        assert scheduler.cycles == 3
        assert clock.slept[:2] == [5.0, 5.0]

    def test_the_interval_is_env_tunable_and_falls_back_when_unparsable(
        self, workspace
    ) -> None:
        assert Scheduler(env=_env(workspace)).interval_seconds == 5.0
        assert Scheduler(env=_env(workspace, WORKER_INTERVAL_SECONDS="1.5")).interval_seconds == 1.5
        assert Scheduler(env=_env(workspace, WORKER_INTERVAL_SECONDS="soon")).interval_seconds == 60.0

    def test_main_builds_a_scheduler_from_the_process_environment(
        self, workspace, monkeypatch
    ) -> None:
        for name, value in _env(workspace).items():
            monkeypatch.setenv(name, value)

        class OneCycle(Scheduler):
            def run_forever(self) -> int:  # type: ignore[override]
                self.run_once()
                return 0

        monkeypatch.setattr("src.workers.scheduler._build_scheduler", OneCycle)
        assert main([]) == 0


class TestPackageExports:
    def test_the_package_resolves_its_public_names_lazily(self) -> None:
        from src import workers

        assert workers.Scheduler is Scheduler
        assert workers.JOB_NAMES == JOB_NAMES
        assert "Scheduler" in dir(workers)

    def test_the_module_is_runnable_as_a_program(self, workspace) -> None:
        """``python -m src.workers.scheduler`` resolves and enters its loop."""
        import os
        import subprocess
        import sys

        child_env = {
            **os.environ,
            **_env(workspace),
            "PYTHONPATH": str(Path.cwd()),
        }
        process = subprocess.Popen(
            [sys.executable, "-m", "src.workers.scheduler"],
            env=child_env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        try:
            _, stderr = process.communicate(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            _, stderr = process.communicate()
            # Still looping after five seconds is the pass condition: the entry
            # point started, found its jobs, and did not crash.
            assert "Traceback" not in stderr
            assert "RuntimeWarning" not in stderr
        else:
            pytest.fail(f"the worker loop exited early: {stderr}")
