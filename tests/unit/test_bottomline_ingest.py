"""Unit tests for Bottomline ingestion pipeline.

Tests cover parsing, transformation, deduplication, quarantine routing,
checksum verification, and the full ingestion orchestration flow.
Uses pytest as the declared test runner.
"""

from __future__ import annotations

import os
import tempfile

import pytest

from src.integration.bottomline.ingestor import BottomlineIngestor
from src.integration.bottomline.models import (
    AuditEntry,
    IngestionStatus,
    compute_checksum,
)
from src.integration.bottomline.parser import BottomlineParseError, parse_bottomline_flat
from src.integration.bottomline.quarantine import quarantine_rejected_file
from src.integration.bottomline.repository import IngestionRepository
from src.integration.bottomline.sftp_poller import SftpFileEntry, SftpPoller
from src.integration.bottomline.transformer import (
    TransformationError,
    transform_to_coa_records,
)


VALID_FLAT_CONTENT = (
    "COA-001|PROD-A|LOT-100|PH|7.2|pH|6.5|8.5|2024-01-15T10:30:00\n"
    "COA-001|PROD-A|LOT-100|VISCOSITY|150|cP|100|200|2024-01-15T10:31:00\n"
)

VALID_FLAT_CONTENT_NO_SPECS = (
    "COA-002|PROD-B|LOT-200|COLOR|Pass||||2024-01-15T11:00:00\n"
)


class TestComputeChecksum:
    """Tests for SHA-256 checksum computation."""

    def test_checksum_deterministic(self) -> None:
        data = b"test content for checksum"
        assert compute_checksum(data) == compute_checksum(data)

    def test_checksum_different_content(self) -> None:
        assert compute_checksum(b"aaa") != compute_checksum(b"bbb")

    def test_checksum_empty(self) -> None:
        result = compute_checksum(b"")
        assert len(result) == 64  # SHA-256 hex digest length


class TestParseBottomlineFlat:
    """Tests for Bottomline flat file parser."""

    def test_parse_valid_content(self) -> None:
        records = parse_bottomline_flat(VALID_FLAT_CONTENT, "BATCH-001")
        assert len(records) == 2
        assert records[0].coa_number == "COA-001"
        assert records[0].test_code == "PH"
        assert records[0].test_result == "7.2"
        assert records[0].specification_min == 6.5
        assert records[0].specification_max == 8.5
        assert records[0].batch_id == "BATCH-001"
        assert records[0].line_number == 1

    def test_parse_without_specs(self) -> None:
        records = parse_bottomline_flat(VALID_FLAT_CONTENT_NO_SPECS, "BATCH-002")
        assert len(records) == 1
        assert records[0].specification_min is None
        assert records[0].specification_max is None
        assert records[0].test_unit == ""

    def test_parse_skips_comments_and_blanks(self) -> None:
        content = "# comment line\n\n" + VALID_FLAT_CONTENT + "\n# another comment"
        records = parse_bottomline_flat(content, "BATCH-001")
        assert len(records) == 2

    def test_parse_empty_content_raises(self) -> None:
        with pytest.raises(BottomlineParseError, match="Empty file content"):
            parse_bottomline_flat("", "BATCH-001")

    def test_parse_only_comments_raises(self) -> None:
        with pytest.raises(BottomlineParseError, match="No valid records"):
            parse_bottomline_flat("# just a comment\n# another", "BATCH-001")

    def test_parse_wrong_field_count_raises(self) -> None:
        with pytest.raises(BottomlineParseError, match="Expected 9 pipe-delimited"):
            parse_bottomline_flat("A|B|C\n", "BATCH-001")

    def test_parse_missing_coa_number_raises(self) -> None:
        bad = "|PROD-A|LOT-100|PH|7.2|pH|6.5|8.5|2024-01-15T10:30:00\n"
        with pytest.raises(BottomlineParseError, match="Missing COA number"):
            parse_bottomline_flat(bad, "BATCH-001")

    def test_parse_missing_test_code_raises(self) -> None:
        bad = "COA-001|PROD-A|LOT-100||7.2|pH|6.5|8.5|2024-01-15T10:30:00\n"
        with pytest.raises(BottomlineParseError, match="Missing test code"):
            parse_bottomline_flat(bad, "BATCH-001")

    def test_parse_invalid_spec_min_raises(self) -> None:
        bad = "COA-001|PROD-A|LOT-100|PH|7.2|pH|abc|8.5|2024-01-15T10:30:00\n"
        with pytest.raises(BottomlineParseError, match="Invalid spec_min"):
            parse_bottomline_flat(bad, "BATCH-001")

    def test_parse_invalid_datetime_raises(self) -> None:
        bad = "COA-001|PROD-A|LOT-100|PH|7.2|pH|6.5|8.5|not-a-date\n"
        with pytest.raises(BottomlineParseError, match="Invalid datetime"):
            parse_bottomline_flat(bad, "BATCH-001")


class TestTransformToCoaRecords:
    """Tests for Bottomline-to-canonical transformation."""

    def test_transform_valid_records(self) -> None:
        raw = parse_bottomline_flat(VALID_FLAT_CONTENT, "BATCH-001")
        items = transform_to_coa_records(raw)
        assert len(items) == 2
        assert items[0].coa_number == "COA-001"
        assert items[0].source_batch_id == "BATCH-001"
        assert items[0].source_line_number == 1

    def test_transform_spec_min_greater_than_max_raises(self) -> None:
        bad = "COA-001|PROD-A|LOT-100|PH|7.2|pH|9.0|8.0|2024-01-15T10:30:00\n"
        raw = parse_bottomline_flat(bad, "BATCH-001")
        with pytest.raises(TransformationError, match="spec_min.*> spec_max"):
            transform_to_coa_records(raw)


class TestIngestionRepository:
    """Tests for the ingestion repository layer."""

    def test_deduplication_marks_seen(self) -> None:
        repo = IngestionRepository()
        assert not repo.is_duplicate("B1", "f.txt")
        repo.mark_seen("B1", "f.txt")
        assert repo.is_duplicate("B1", "f.txt")

    def test_deduplication_different_files(self) -> None:
        repo = IngestionRepository()
        repo.mark_seen("B1", "f1.txt")
        assert not repo.is_duplicate("B1", "f2.txt")
        assert not repo.is_duplicate("B2", "f1.txt")

    def test_save_and_get_records(self) -> None:
        repo = IngestionRepository()
        raw = parse_bottomline_flat(VALID_FLAT_CONTENT, "BATCH-001")
        items = transform_to_coa_records(raw)
        repo.save_records("BATCH-001", items)
        retrieved = repo.get_records("BATCH-001")
        assert len(retrieved) == 2

    def test_audit_entries(self) -> None:
        repo = IngestionRepository()
        entry = AuditEntry(
            batch_id="B1",
            file_name="f.txt",
            action="TEST",
            status="OK",
        )
        repo.add_audit_entry(entry)
        entries = repo.get_audit_entries("B1")
        assert len(entries) == 1
        assert entries[0].action == "TEST"

    def test_audit_entries_filter(self) -> None:
        repo = IngestionRepository()
        repo.add_audit_entry(
            AuditEntry(batch_id="B1", file_name="f.txt", action="A", status="OK")
        )
        repo.add_audit_entry(
            AuditEntry(batch_id="B2", file_name="g.txt", action="B", status="OK")
        )
        assert len(repo.get_audit_entries("B1")) == 1
        assert len(repo.get_audit_entries()) == 2


class TestQuarantine:
    """Tests for error quarantine routing."""

    def test_quarantine_writes_file(self) -> None:
        repo = IngestionRepository()
        with tempfile.TemporaryDirectory() as tmpdir:
            result = quarantine_rejected_file(
                batch_id="B1",
                file_name="bad_file.txt",
                content=b"corrupt data",
                reason="Parse failure",
                repository=repo,
                quarantine_dir=tmpdir,
            )
            assert os.path.exists(result.quarantine_path)
            assert result.reason == "Parse failure"
            assert "B1" in result.quarantine_path

    def test_quarantine_creates_audit_entry(self) -> None:
        repo = IngestionRepository()
        with tempfile.TemporaryDirectory() as tmpdir:
            quarantine_rejected_file(
                batch_id="B1",
                file_name="bad.txt",
                content=b"data",
                reason="Test rejection",
                repository=repo,
                quarantine_dir=tmpdir,
            )
            entries = repo.get_audit_entries("B1")
            assert len(entries) == 1
            assert entries[0].action == "QUARANTINE"
            assert entries[0].status == "REJECTED"

    def test_quarantine_safe_filename(self) -> None:
        """Ensure path traversal attempts are neutralized via Path.name."""
        repo = IngestionRepository()
        with tempfile.TemporaryDirectory() as tmpdir:
            result = quarantine_rejected_file(
                batch_id="B1",
                file_name="../../../etc/passwd",
                content=b"malicious",
                reason="Path traversal attempt",
                repository=repo,
                quarantine_dir=tmpdir,
            )
            assert "passwd" in result.quarantine_path
            assert ".." not in result.quarantine_path


class TestBottomlineIngestor:
    """Integration tests for the full ingestion orchestrator."""

    def _make_entry(self, content: str = VALID_FLAT_CONTENT) -> SftpFileEntry:
        return SftpFileEntry(
            file_name="BATCH_B001_001.txt",
            content=content.encode("utf-8"),
            batch_id="B001",
        )

    def test_successful_ingestion(self) -> None:
        repo = IngestionRepository()
        ingestor = BottomlineIngestor(repository=repo)
        entry = self._make_entry()

        result = ingestor.ingest_file(entry)

        assert result.status == IngestionStatus.PERSISTED
        assert result.record_count == 2
        assert result.batch_id == "B001"
        assert len(repo.get_records("B001")) == 2

    def test_duplicate_detection_is_idempotent(self) -> None:
        repo = IngestionRepository()
        ingestor = BottomlineIngestor(repository=repo)
        entry = self._make_entry()

        first = ingestor.ingest_file(entry)
        second = ingestor.ingest_file(entry)

        assert first.status == IngestionStatus.PERSISTED
        assert second.status == IngestionStatus.DUPLICATE
        assert len(repo.get_records("B001")) == 2  # Not duplicated

    def test_parse_failure_routes_to_quarantine(self) -> None:
        repo = IngestionRepository()
        with tempfile.TemporaryDirectory() as tmpdir:
            ingestor = BottomlineIngestor(
                repository=repo, quarantine_dir=tmpdir
            )
            entry = SftpFileEntry(
                file_name="BATCH_BAD_001.txt",
                content=b"invalid|data\n",
                batch_id="BAD",
            )

            result = ingestor.ingest_file(entry)

            assert result.status == IngestionStatus.QUARANTINED
            assert result.error_reason is not None
            assert "Parse error" in result.error_reason

    def test_transformation_failure_routes_to_quarantine(self) -> None:
        repo = IngestionRepository()
        with tempfile.TemporaryDirectory() as tmpdir:
            ingestor = BottomlineIngestor(
                repository=repo, quarantine_dir=tmpdir
            )
            bad_content = (
                "COA-001|PROD-A|LOT-100|PH|7.2|pH|9.0|8.0|2024-01-15T10:30:00\n"
            )
            entry = SftpFileEntry(
                file_name="BATCH_TXERR_001.txt",
                content=bad_content.encode("utf-8"),
                batch_id="TXERR",
            )

            result = ingestor.ingest_file(entry)

            assert result.status == IngestionStatus.QUARANTINED
            assert "Transformation error" in (result.error_reason or "")

    def test_checksum_recorded_in_audit(self) -> None:
        repo = IngestionRepository()
        ingestor = BottomlineIngestor(repository=repo)
        entry = self._make_entry()

        ingestor.ingest_file(entry)

        audit_entries = repo.get_audit_entries("B001")
        start_entries = [e for e in audit_entries if e.action == "INGEST_START"]
        assert len(start_entries) == 1
        assert "checksum" in start_entries[0].details

    def test_ingestion_with_no_specs(self) -> None:
        repo = IngestionRepository()
        ingestor = BottomlineIngestor(repository=repo)
        entry = SftpFileEntry(
            file_name="BATCH_NOSPEC_001.txt",
            content=VALID_FLAT_CONTENT_NO_SPECS.encode("utf-8"),
            batch_id="NOSPEC",
        )

        result = ingestor.ingest_file(entry)
        assert result.status == IngestionStatus.PERSISTED
        assert result.record_count == 1


class MockSftpClient:
    """Mock SFTP client for testing the poller."""

    def __init__(self, files: dict[str, bytes]) -> None:
        self._files = files
        self._deleted: list[str] = []

    def list_files(self, remote_dir: str) -> list[str]:
        return list(self._files.keys())

    def read_file(self, remote_path: str) -> bytes:
        name = remote_path.split("/")[-1]
        if name in self._files:
            return self._files[name]
        raise FileNotFoundError(remote_path)

    def delete_file(self, remote_path: str) -> None:
        self._deleted.append(remote_path)


class TestSftpPoller:
    """Tests for the SFTP polling mechanism."""

    def test_poll_discovers_files(self) -> None:
        client = MockSftpClient(
            {"BATCH_001_001.txt": b"data1", "BATCH_002_001.txt": b"data2"}
        )
        poller = SftpPoller(client=client, remote_dir="/inbound")
        entries = poller.poll()
        assert len(entries) == 2
        assert entries[0].batch_id == "001"
        assert entries[1].batch_id == "002"

    def test_poll_filters_non_matching(self) -> None:
        client = MockSftpClient(
            {"BATCH_001_001.txt": b"data", "readme.md": b"info"}
        )
        poller = SftpPoller(client=client, remote_dir="/inbound")
        entries = poller.poll()
        assert len(entries) == 1

    def test_acknowledge_deletes_file(self) -> None:
        client = MockSftpClient({"test.txt": b"data"})
        poller = SftpPoller(client=client, remote_dir="/inbound")
        poller.acknowledge("test.txt")
        assert "/inbound/test.txt" in client._deleted
