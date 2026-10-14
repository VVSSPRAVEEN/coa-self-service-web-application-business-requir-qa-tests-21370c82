"""FR11: the email-only notification dispatcher.

:mod:`src.notifications.email` already exists, so this module tests the seams it
actually exposes rather than a parallel implementation:

* **channel guard** - anything that is not ``email`` is rejected before a single
  message is built, and the rejection is logged;
* **retry** - a transport that keeps failing is retried up to ``max_retries``
  times with bounded exponential backoff, and the failure count is exact;
* **idempotency** - the same report window is delivered to the same recipient
  once, however many times ``dispatch`` is called;
* **dry run** - nothing reaches the transport, but the dispatch is still recorded
  so a dry run cannot be followed by a real send of the same window.

Backoff delays are set to zero in the fixtures so the retry tests assert the
*attempt count* rather than the wall-clock cost of sleeping.
"""

from __future__ import annotations

import base64
from typing import Any, Optional

import pytest

from src.notifications.email import (
    EmailDispatcher,
    MockMailTransport,
    SmtpMailTransport,
)
from src.notifications.email.dispatcher import (
    ALLOWED_CHANNELS,
    BLOCKED_CHANNELS,
    ChannelGuardError,
    DispatcherSettings,
    IdempotencyStore,
)
from src.notifications.email.transport import MailTransport, build_mime_message

SENDER = "no-reply@ventura.example"
CS = ["cs@ventura.example"]
QA = ["qa@ventura.example"]


class CountingTransport:
    """Transport that fails a configurable number of times before succeeding."""

    def __init__(self, failures: int = 0) -> None:
        self.failures = failures
        self.attempts = 0
        self.sent: list[dict[str, Any]] = []

    def send(
        self,
        sender: str,
        recipients: list[str],
        subject: str,
        body_text: str,
        body_html: Optional[str] = None,
        attachments: Optional[list[dict[str, Any]]] = None,
    ) -> bool:
        self.attempts += 1
        if self.attempts <= self.failures:
            return False
        self.sent.append(
            {
                "sender": sender,
                "recipients": list(recipients),
                "subject": subject,
                "body_text": body_text,
                "attachments": attachments or [],
            }
        )
        return True


def _settings(**overrides: Any) -> DispatcherSettings:
    defaults: dict[str, Any] = {
        "sender_address": SENDER,
        "cs_recipients": tuple(CS),
        "qa_recipients": tuple(QA),
        "max_retries": 3,
        "retry_base_delay": 0.0,
        "retry_max_delay": 0.0,
    }
    defaults.update(overrides)
    return DispatcherSettings(**defaults)


def _dispatcher(transport: MailTransport, **overrides: Any) -> EmailDispatcher:
    return EmailDispatcher(transport, _settings(**overrides))


def _dispatch(dispatcher: EmailDispatcher, **overrides: Any) -> bool:
    kwargs: dict[str, Any] = {
        "report_window": "2026-03-31-daily",
        "subject": "Missing COA report",
        "body_text": "One COA is blocking a shipment.",
        "csv_content": "coa_id,vessel\nCOA-1,MV Ventura\n",
    }
    kwargs.update(overrides)
    return dispatcher.dispatch(**kwargs)


class TestChannelGuard:
    def test_email_is_the_only_allowed_channel(self) -> None:
        assert ALLOWED_CHANNELS == frozenset({"email"})

    @pytest.mark.parametrize("channel", sorted(BLOCKED_CHANNELS))
    def test_a_blocked_channel_is_rejected(self, channel: str) -> None:
        with pytest.raises(ChannelGuardError):
            _dispatch(_dispatcher(MockMailTransport()), channel=channel)

    def test_an_unknown_channel_is_rejected(self) -> None:
        with pytest.raises(ChannelGuardError):
            _dispatch(_dispatcher(MockMailTransport()), channel="carrier-pigeon")

    def test_the_guard_runs_before_anything_is_sent(self) -> None:
        transport = MockMailTransport()

        with pytest.raises(ChannelGuardError):
            _dispatch(_dispatcher(transport), channel="slack")

        assert transport.sent_messages == []

    @pytest.mark.parametrize("channel", ["email", "EMAIL", " Email "])
    def test_the_guard_normalises_its_input(self, channel: str) -> None:
        assert _dispatch(_dispatcher(MockMailTransport()), channel=channel) is True

    def test_enforce_channel_is_usable_on_its_own(self) -> None:
        dispatcher = _dispatcher(MockMailTransport())

        dispatcher.enforce_channel("email")

        with pytest.raises(ChannelGuardError):
            dispatcher.enforce_channel("sms")


class TestRecipients:
    def test_both_lists_are_addressed_by_default(self) -> None:
        assert _dispatcher(MockMailTransport()).resolve_recipients() == sorted(CS + QA)

    def test_a_named_list_selects_only_that_one(self) -> None:
        assert _dispatcher(MockMailTransport()).resolve_recipients(["cs"]) == CS

    def test_recipients_are_deduplicated(self) -> None:
        shared = ("ops@ventura.example",)
        dispatcher = _dispatcher(MockMailTransport(), cs_recipients=shared, qa_recipients=shared)

        assert dispatcher.resolve_recipients() == ["ops@ventura.example"]

    def test_every_recipient_gets_the_attachment(self) -> None:
        transport = MockMailTransport()

        _dispatch(_dispatcher(transport))

        assert len(transport.sent_messages) == 2
        for message in transport.sent_messages:
            assert message["attachments"][0]["filename"] == (
                "missing-coa-report-2026-03-31-daily.csv"
            )
            assert message["attachments"][0]["content"] == (
                b"coa_id,vessel\nCOA-1,MV Ventura\n"
            )

    def test_a_dispatch_with_no_recipients_succeeds_silently(self, caplog) -> None:
        dispatcher = _dispatcher(MockMailTransport(), cs_recipients=(), qa_recipients=())

        with caplog.at_level("WARNING"):
            assert _dispatch(dispatcher) is True

        assert "No recipients resolved" in caplog.text


class TestRetry:
    def test_a_transient_failure_is_retried_until_it_succeeds(self) -> None:
        transport = CountingTransport(failures=2)

        assert _dispatch(_dispatcher(transport), distribution_lists=["cs"]) is True
        assert transport.attempts == 3
        assert len(transport.sent) == 1

    def test_retries_stop_at_max_retries_and_report_failure(self) -> None:
        transport = CountingTransport(failures=99)

        assert _dispatch(_dispatcher(transport), distribution_lists=["cs"]) is False
        assert transport.attempts == 3

    def test_the_first_attempt_is_not_a_retry(self) -> None:
        transport = CountingTransport(failures=0)

        _dispatch(_dispatcher(transport), distribution_lists=["cs"])

        assert transport.attempts == 1

    def test_one_permanently_failing_recipient_fails_the_whole_dispatch(self) -> None:
        dispatcher = _dispatcher(CountingTransport(failures=99))

        assert _dispatch(dispatcher) is False

    def test_a_single_retry_is_honoured(self) -> None:
        transport = CountingTransport(failures=99)

        _dispatch(_dispatcher(transport, max_retries=1), distribution_lists=["cs"])

        assert transport.attempts == 1


class TestIdempotency:
    def test_the_key_is_deterministic_for_a_window_and_recipient(self) -> None:
        dispatcher = _dispatcher(MockMailTransport())

        first = dispatcher.generate_idempotency_key("2026-03-31-daily", CS[0])
        second = dispatcher.generate_idempotency_key("2026-03-31-daily", CS[0])

        assert first == second
        assert len(first) == 64

    def test_a_different_window_yields_a_different_key(self) -> None:
        dispatcher = _dispatcher(MockMailTransport())

        assert dispatcher.generate_idempotency_key("day-a", CS[0]) != (
            dispatcher.generate_idempotency_key("day-b", CS[0])
        )

    def test_a_redispatch_of_the_same_window_sends_nothing(self) -> None:
        transport = MockMailTransport()
        dispatcher = _dispatcher(transport, cs_recipients=tuple(CS), qa_recipients=())

        assert _dispatch(dispatcher) is True
        assert _dispatch(dispatcher) is True
        assert len(transport.sent_messages) == 1

    def test_a_new_window_sends_again(self) -> None:
        transport = MockMailTransport()
        dispatcher = _dispatcher(transport, cs_recipients=tuple(CS), qa_recipients=())

        _dispatch(dispatcher)
        _dispatch(dispatcher, report_window="2026-04-01-daily")

        assert len(transport.sent_messages) == 2

    def test_the_store_reports_what_it_has_seen(self) -> None:
        store = IdempotencyStore()

        assert store.has_been_sent("k") is False
        store.mark_sent("k")
        assert store.has_been_sent("k") is True
        store.clear()
        assert store.has_been_sent("k") is False

    def test_a_failed_dispatch_is_not_marked_as_sent(self) -> None:
        store = IdempotencyStore()
        transport = CountingTransport(failures=99)
        dispatcher = EmailDispatcher(
            transport,
            _settings(cs_recipients=tuple(CS), qa_recipients=()),
            idempotency_store=store,
        )

        _dispatch(dispatcher)

        assert store.has_been_sent(dispatcher.generate_idempotency_key("2026-03-31-daily", CS[0])) is False


class TestDryRun:
    def test_a_dry_run_sends_nothing_and_still_succeeds(self) -> None:
        transport = MockMailTransport()

        assert _dispatch(_dispatcher(transport, dry_run=True)) is True
        assert transport.sent_messages == []

    def test_a_dry_run_still_claims_the_window_so_it_cannot_then_be_sent(self) -> None:
        transport = MockMailTransport()
        dispatcher = _dispatcher(transport, dry_run=True, cs_recipients=tuple(CS), qa_recipients=())

        _dispatch(dispatcher)
        _dispatch(dispatcher)

        assert transport.sent_messages == []


class TestTransportBoundary:
    def test_the_mock_transport_records_what_it_was_given(self) -> None:
        transport = MockMailTransport()

        assert transport.send(SENDER, CS, "subject", "body") is True
        assert transport.sent_messages[0]["subject"] == "subject"

    def test_the_mock_transport_can_be_told_to_fail(self) -> None:
        transport = MockMailTransport()
        transport.set_should_fail(True)

        assert transport.send(SENDER, CS, "subject", "body") is False
        assert transport.sent_messages == []

    def test_the_mime_message_is_assembled_with_the_expected_headers(self) -> None:
        raw = build_mime_message(SENDER, CS, "Missing COA", "plain body")

        assert raw.split("\r\n")[0].startswith("Date: ")
        assert f"From: {SENDER}" in raw
        assert f"To: {CS[0]}" in raw
        assert "Subject: Missing COA" in raw
        assert "MIME-Version: 1.0" in raw
        assert "Message-ID: <" in raw
        assert "Content-Type: text/plain; charset=UTF-8" in raw

    def test_the_body_is_base64_transport_encoded(self) -> None:
        raw = build_mime_message(SENDER, CS, "Missing COA", "plain body")

        assert base64.b64encode(b"plain body").decode("ascii") in raw

    def test_the_mime_message_carries_an_attachment_part(self) -> None:
        raw = build_mime_message(
            SENDER,
            CS,
            "Missing COA",
            "plain body",
            attachments=[{"filename": "report.csv", "content": "a,b\n1,2\n"}],
        )

        assert "multipart/mixed" in raw
        assert 'filename="report.csv"' in raw

    def test_an_html_body_produces_a_multipart_alternative(self) -> None:
        raw = build_mime_message(SENDER, CS, "Missing COA", "plain", body_html="<p>rich</p>")

        assert "multipart/alternative" in raw
        assert "text/html; charset=UTF-8" in raw
        assert base64.b64encode(b"<p>rich</p>").decode("ascii") in raw

    def test_a_non_ascii_subject_is_rfc2047_encoded(self) -> None:
        raw = build_mime_message(SENDER, CS, "Bericht für Ventura", "body")

        assert "=?UTF-8?B?" in raw

    def test_the_smtp_transport_reports_an_unreachable_server_as_a_failure(self) -> None:
        """A dead relay must come back as ``False``, never as an exception."""

        transport = SmtpMailTransport("127.0.0.1", 1, timeout=1.0)

        assert transport.send(SENDER, CS, "Missing COA", "body") is False

    def test_an_unexpected_transport_exception_is_not_swallowed(self) -> None:
        """The retry loop does not catch arbitrary errors, so a bug stays visible."""

        class ExplodingTransport:
            def send(self, *args: Any, **kwargs: Any) -> bool:
                raise RuntimeError("the mail server exploded")

        dispatcher = _dispatcher(ExplodingTransport(), cs_recipients=tuple(CS), qa_recipients=())

        with pytest.raises(RuntimeError):
            _dispatch(dispatcher)
