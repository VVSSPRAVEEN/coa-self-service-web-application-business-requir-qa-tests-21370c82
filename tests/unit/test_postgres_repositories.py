"""The durable PostgreSQL repositories, driven without a database server.

Item 1 of the runtime review: three persistence boundaries shipped as in-memory
dicts, a JSON file and a memory-backed audit trail, each with a "would be
PostgreSQL in production" note in place of a store. They now have real
PostgreSQL implementations, and this module is what proves they are real rather
than merely declared:

* :class:`RecordingConnection` records every statement the repositories emit and
  answers each one from canned rows, so a test asserts the exact SQL and the
  exact parameters without a server. :class:`AuditTrailDatabase` goes further and
  interprets the audit statements well enough to *accumulate* rows, which lets
  :class:`~src.audit.trail.service.AuditTrailService` record, read back and
  verify a chain across a "durable" backend rather than in memory.
* :class:`TestSchemaCoherence` pins every column name the three repositories
  name against ``migrations/001_initial_schema.sql``. That is the guard against
  the failure mode a canned-response test cannot see: SQL that is internally
  consistent but names a column the migration never creates.
* :class:`TestLiveRoundTrip` runs the same repositories against a real server
  when one is reachable, and skips cleanly - never fails - when it is not.

Nothing here needs ``psycopg2`` to be installed to exercise the SQL: the driver
is imported lazily, only when a DSN string (rather than an injected connection)
is handed to :func:`src.core.postgres.connection`.
"""

from __future__ import annotations

import json
import os
import re
import uuid
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Self

import pytest

from src.admin.plant_mapping.models import PlantMappingCreate, PlantMappingUpdate
from src.admin.plant_mapping.repository import (
    PlantMappingRepository,
    PostgresPlantMappingRepository,
    build_plant_mapping_repository,
)
from src.admin.plant_mapping.service import DuplicateMappingError, PlantMappingService
from src.admin.users import (
    DuplicateUserError,
    PostgresUserRegistry,
    UserDTO,
    UserNotFoundError,
    _store,
    create_external_account,
    get_user,
    list_users,
    update_role,
)
from src.audit.trail.models import AuditEventType
from src.audit.trail.repository import (
    AuditRepository,
    InMemoryAuditStorage,
    PostgresAuditStorage,
    build_audit_storage,
)
from src.audit.trail.service import AuditTrailService
from src.core.postgres import (
    BACKEND_MEMORY,
    BACKEND_POSTGRES,
    PersistenceConfigurationError,
    connection,
    resolve_backend,
    resolve_dsn,
)

DSN = "postgresql://coa:coa_local_only@audit-db:5432/coa_audit"

MIGRATION = (
    Path(__file__).resolve().parents[2] / "migrations" / "001_initial_schema.sql"
)

MIGRATION_SQL = MIGRATION.read_text(encoding="utf-8")


@pytest.fixture(name="migration")
def migration_fixture() -> str:
    """The checked-in schema DDL every coherence assertion reads."""
    return MIGRATION_SQL


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


def _normalize(sql: str) -> str:
    """Collapse whitespace so a matcher can span a wrapped statement."""
    return re.sub(r"\s+", " ", sql).strip()


class FakeCursor:
    """A DB-API cursor that answers from the connection's canned responses."""

    def __init__(self, connection: RecordingConnection) -> None:
        self._connection = connection
        self._rows: list[Any] = []
        self.closed = False

    def execute(self, sql: str, params: Sequence[Any] = ()) -> FakeCursor:
        self._connection.statements.append((_normalize(sql), tuple(params)))
        self._rows = self._connection.respond(sql, params)
        return self

    def fetchone(self) -> Any:
        return self._rows.pop(0) if self._rows else None

    def fetchall(self) -> list[Any]:
        rows, self._rows = list(self._rows), []
        return rows

    def close(self) -> None:
        self.closed = True

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()


class RecordingConnection:
    """Records emitted SQL and answers it from substring-matched responses.

    Args:
        responses: ``(matcher, rows)`` pairs. The first pair whose matcher is a
            substring of the normalised statement supplies the rows for that
            statement; a statement matching nothing yields no rows, which is how
            a plain ``INSERT`` (no ``RETURNING``) is modelled.
    """

    def __init__(
        self,
        responses: Sequence[tuple[str, Sequence[Any]]] = (),
    ) -> None:
        self.statements: list[tuple[str, tuple[Any, ...]]] = []
        self.commits = 0
        self.rollbacks = 0
        self.closed = False
        self._responses = list(responses)
        self.cursors: list[FakeCursor] = []

    def cursor(self) -> FakeCursor:
        cursor = FakeCursor(self)
        self.cursors.append(cursor)
        return cursor

    def respond(self, sql: str, params: Sequence[Any]) -> list[Any]:
        normalized = _normalize(sql)
        for matcher, rows in self._responses:
            if _normalize(matcher) in normalized:
                return list(rows)
        return []

    def statement_matching(self, fragment: str) -> tuple[str, tuple[Any, ...]] | None:
        """Return the first recorded statement containing ``fragment``."""
        needle = _normalize(fragment)
        for sql, params in self.statements:
            if needle in sql:
                return sql, params
        return None

    def statements_matching(self, fragment: str) -> list[tuple[str, tuple[Any, ...]]]:
        """Return every recorded statement containing ``fragment``."""
        needle = _normalize(fragment)
        return [(sql, params) for sql, params in self.statements if needle in sql]

    def commit(self) -> None:
        self.commits += 1

    def rollback(self) -> None:
        self.rollbacks += 1

    def close(self) -> None:
        self.closed = True


class PlantMappingDatabase(RecordingConnection):
    """A recording connection that also accumulates ``plant_mappings`` rows.

    It interprets the statements the FR9 repository depends on - the insert, the
    reads, the update and the delete - so a full create/read/update/delete
    lifecycle can be asserted against the durable store the way it is asserted
    against the in-memory one.
    """

    _COLUMNS = (
        "mapping_uuid",
        "source_plant_id",
        "target_plant_id",
        "active",
        "created_by",
        "updated_by",
        "created_at",
        "updated_at",
    )

    def __init__(self) -> None:
        super().__init__()
        self.rows: list[dict[str, Any]] = []

    def respond(self, sql: str, params: Sequence[Any]) -> list[Any]:
        normalized = _normalize(sql)
        if normalized.startswith("INSERT INTO plant_mappings"):
            row = dict(zip(self._COLUMNS, params))
            row["created_at"] = row["created_at"]
            self.rows.append(row)
            return [self.rows[-1]]
        if normalized.startswith("UPDATE plant_mappings"):
            row = self._find(params[4])
            if row is None:
                return []
            if params[0] is not None:
                row["target_plant_id"] = params[0]
            if params[1] is not None:
                row["active"] = params[1]
            row["updated_by"] = params[2]
            row["updated_at"] = params[3]
            return [row]
        if normalized.startswith("DELETE FROM plant_mappings"):
            row = self._find(params[0])
            if row is None:
                return []
            self.rows.remove(row)
            return [row["mapping_uuid"]]
        if "WHERE mapping_uuid" in normalized:
            row = self._find(params[0])
            return [] if row is None else [row]
        if "WHERE source_plant_id" in normalized:
            return [row for row in self.rows if row["source_plant_id"] == params[0]]
        if "FROM plant_mappings" in normalized:
            return list(self.rows)
        return []

    def _find(self, mapping_uuid: Any) -> dict[str, Any] | None:
        return next(
            (row for row in self.rows if row["mapping_uuid"] == mapping_uuid), None
        )


class AuditTrailDatabase(RecordingConnection):
    """A recording connection that also accumulates ``audit_events`` rows.

    It interprets the three statement shapes the audit storage depends on -
    the advisory lock, the next-sequence lookup and the insert - and answers the
    reads back from what it accumulated, ordered by ``sequence_number`` exactly
    as the real ``ORDER BY sequence_number`` does. That is what makes a
    service-level round trip meaningful instead of a canned-response assertion.
    """

    _INSERT_COLUMNS = (
        "event_id",
        "external_event_id",
        "event_type",
        "actor_id",
        "coa_id",
        "occurred_at",
        "payload_json",
        "prev_hash",
        "entry_hash",
        "sequence_number",
        "details",
        "ip_address",
    )

    def __init__(self) -> None:
        super().__init__()
        self.rows: list[dict[str, Any]] = []

    def respond(self, sql: str, params: Sequence[Any]) -> list[Any]:
        normalized = _normalize(sql)
        if "pg_advisory_xact_lock" in normalized:
            return []
        if "MAX(sequence_number)" in normalized:
            highest = max((row["sequence_number"] for row in self.rows), default=-1)
            return [{"next_sequence": highest + 1}]
        if normalized.startswith("INSERT INTO audit_events"):
            self.rows.append(dict(zip(self._INSERT_COLUMNS, params)))
            return []
        if "count(*)" in normalized:
            return [{"entry_count": len(self.rows)}]
        if "FROM audit_events" in normalized:
            return self._select(normalized)
        return []

    def _select(self, normalized: str) -> list[dict[str, Any]]:
        matched = list(self.rows)
        where = ""
        if "WHERE coa_id =" in normalized:
            where = "coa_id"
            value = self.statements[-1][1][0]
            matched = [row for row in matched if row["coa_id"] == value]
        elif "WHERE event_type =" in normalized:
            where = "event_type"
            value = self.statements[-1][1][0]
            matched = [row for row in matched if row["event_type"] == value]
        elif "WHERE occurred_at <" in normalized:
            where = "occurred_at"
            value = self.statements[-1][1][0]
            matched = [row for row in matched if row["occurred_at"] < value]
        assert where or normalized.endswith("ORDER BY sequence_number")
        return sorted(matched, key=lambda row: row["sequence_number"])


def _mapping_row(
    mapping_uuid: str = "mapping-1",
    *,
    sister: str = "PLANT-B",
    active: bool = True,
    updated_by: str | None = "admin-user-1",
) -> dict[str, Any]:
    """One ``plant_mappings`` row in the shape the repository reads."""
    return {
        "mapping_uuid": mapping_uuid,
        "source_plant_id": "PLANT-A",
        "target_plant_id": sister,
        "active": active,
        "created_by": "admin-user-1",
        "updated_by": updated_by,
        "created_at": datetime(2026, 3, 1, 12, 0, tzinfo=UTC),
        "updated_at": datetime(2026, 3, 2, 9, 30, tzinfo=UTC),
    }


def _user_row(user_id: str = "u-1", username: str = "dana.ops") -> dict[str, Any]:
    """One ``admin_users`` row in the shape the registry reads."""
    return {
        "user_id": user_id,
        "username": username,
        "email": "dana.ops@example.com",
        "role": "3pl",
        "organization": "Harbor Freight",
        "created_at": datetime(2026, 3, 1, 12, 0, tzinfo=UTC),
    }


class _UniqueViolation(Exception):
    """Stands in for psycopg2's UniqueViolation, which carries a SQLSTATE."""

    pgcode = "23505"


# ---------------------------------------------------------------------------
# Backend selection
# ---------------------------------------------------------------------------


class TestBackendSelection:
    def test_no_dsn_selects_the_documented_in_memory_fallback(self) -> None:
        assert resolve_backend({}) == BACKEND_MEMORY
        assert resolve_dsn({}) is None

    @pytest.mark.parametrize("name", ["P3_AUDIT_DSN", "DATABASE_URL"])
    def test_either_dsn_variable_selects_postgres(self, name: str) -> None:
        assert resolve_backend({name: DSN}) == BACKEND_POSTGRES

    def test_p3_audit_dsn_wins_over_the_conventional_alias(self) -> None:
        env = {"P3_AUDIT_DSN": DSN, "DATABASE_URL": "postgresql://other/db"}
        assert resolve_dsn(env) == DSN

    def test_a_blank_value_is_not_a_dsn(self) -> None:
        assert resolve_backend({"P3_AUDIT_DSN": "   "}) == BACKEND_MEMORY

    def test_the_fallback_can_be_pinned_even_with_a_dsn_present(self) -> None:
        env = {"P3_AUDIT_DSN": DSN, "P3_PERSISTENCE_BACKEND": "memory"}
        assert resolve_backend(env) == BACKEND_MEMORY

    def test_pinning_postgres_without_a_dsn_is_an_error_not_a_degradation(self) -> None:
        with pytest.raises(PersistenceConfigurationError):
            resolve_backend({"P3_PERSISTENCE_BACKEND": "postgres"})

    def test_an_unknown_backend_name_is_rejected(self) -> None:
        with pytest.raises(ValueError):
            resolve_backend({"P3_PERSISTENCE_BACKEND": "sqlite"})


class TestFactories:
    def test_plant_mapping_factory_defaults_to_the_in_memory_store(self) -> None:
        assert isinstance(build_plant_mapping_repository({}), PlantMappingRepository)

    def test_plant_mapping_factory_uses_postgres_when_a_dsn_is_configured(self) -> None:
        built = build_plant_mapping_repository({"P3_AUDIT_DSN": DSN})
        assert isinstance(built, PostgresPlantMappingRepository)

    def test_plant_mapping_factory_honours_an_explicitly_pinned_fallback(self) -> None:
        env = {"P3_AUDIT_DSN": DSN, "P3_PERSISTENCE_BACKEND": "memory"}
        assert isinstance(build_plant_mapping_repository(env), PlantMappingRepository)

    def test_audit_storage_factory_defaults_to_the_in_memory_store(self) -> None:
        assert isinstance(build_audit_storage({}), InMemoryAuditStorage)

    def test_audit_storage_factory_uses_postgres_when_a_dsn_is_configured(self) -> None:
        assert isinstance(build_audit_storage({"P3_AUDIT_DSN": DSN}), PostgresAuditStorage)

    def test_the_repository_default_is_the_selected_backend(self, monkeypatch) -> None:
        monkeypatch.setenv("P3_AUDIT_DSN", DSN)
        monkeypatch.delenv("DATABASE_URL", raising=False)
        assert isinstance(AuditRepository()._storage, PostgresAuditStorage)

    def test_the_registry_default_is_the_in_memory_store_without_a_dsn(
        self, monkeypatch
    ) -> None:
        monkeypatch.delenv("P3_AUDIT_DSN", raising=False)
        monkeypatch.delenv("DATABASE_URL", raising=False)
        assert not isinstance(_store(), PostgresUserRegistry)

    def test_the_registry_default_is_postgres_when_a_dsn_is_configured(
        self, monkeypatch
    ) -> None:
        monkeypatch.setenv("P3_AUDIT_DSN", DSN)
        assert isinstance(_store(), PostgresUserRegistry)


class TestInjectedConnectionsAreNotOwned:
    def test_an_injected_connection_is_neither_committed_nor_closed(self) -> None:
        fake = RecordingConnection()
        with connection(fake) as conn:
            assert conn is fake
        assert fake.commits == 0
        assert fake.rollbacks == 0

    def test_a_dsn_owns_the_connection_it_opens(self, monkeypatch) -> None:
        opened: list[RecordingConnection] = []

        def fake_connect(dsn: str) -> RecordingConnection:
            conn = RecordingConnection()
            opened.append(conn)
            return conn

        monkeypatch.setattr("psycopg2.connect", fake_connect, raising=False)
        with connection(DSN) as conn:
            conn.cursor()
        assert opened[0].commits == 1
        assert opened[0].closed is True


# ---------------------------------------------------------------------------
# FR9 plant mappings
# ---------------------------------------------------------------------------


class TestPostgresPlantMappingRepository:
    def test_create_inserts_the_model_id_and_returns_the_stored_row(self) -> None:
        fake = RecordingConnection([("RETURNING", [_mapping_row()])])
        repo = PostgresPlantMappingRepository(fake)

        stored = repo.create_sync(
            PlantMappingCreate(
                source_plant_id="PLANT-A", sister_plant_id="PLANT-B", created_by="admin-user-1"
            )
        )

        statement = fake.statement_matching("INSERT INTO plant_mappings")
        assert statement is not None
        sql, params = statement
        assert "mapping_uuid" in sql and "target_plant_id" in sql
        # The id is the model's own factory, so it is a parseable uuid4 string.
        assert uuid.UUID(params[0])
        assert params[1:6] == ("PLANT-A", "PLANT-B", True, "admin-user-1", "admin-user-1")
        assert stored.id == "mapping-1"
        assert stored.sister_plant_id == "PLANT-B"
        assert stored.is_active is True

    def test_get_by_id_returns_none_when_the_row_is_unknown(self) -> None:
        repo = PostgresPlantMappingRepository(RecordingConnection())
        assert repo.get_by_id_sync("nope") is None

    def test_list_all_reads_in_insertion_order(self) -> None:
        fake = RecordingConnection(
            [("ORDER BY id", [_mapping_row("mapping-1"), _mapping_row("mapping-2")])]
        )
        stored = PostgresPlantMappingRepository(fake).list_all_sync()

        assert [mapping.id for mapping in stored] == ["mapping-1", "mapping-2"]

    def test_get_by_source_plant_filters_in_the_database(self) -> None:
        fake = RecordingConnection([("WHERE source_plant_id", [_mapping_row()])])
        PostgresPlantMappingRepository(fake).get_by_source_plant_sync("PLANT-A")

        statement = fake.statement_matching("WHERE source_plant_id")
        assert statement is not None and statement[1] == ("PLANT-A",)

    def test_update_coalesces_the_fields_the_caller_left_unset(self) -> None:
        fake = RecordingConnection([("UPDATE plant_mappings", [_mapping_row(sister="PLANT-C")])])
        stored = PostgresPlantMappingRepository(fake).update_sync(
            "mapping-1", PlantMappingUpdate(sister_plant_id="PLANT-C", updated_by="admin-2")
        )

        sql, params = fake.statement_matching("UPDATE plant_mappings")
        assert sql.count("COALESCE") == 2
        assert params[0] == "PLANT-C"
        assert params[1] is None
        assert stored is not None and stored.sister_plant_id == "PLANT-C"

    def test_update_of_an_unknown_mapping_is_none_not_an_error(self) -> None:
        repo = PostgresPlantMappingRepository(RecordingConnection())
        assert repo.update_sync("nope", PlantMappingUpdate(updated_by="a")) is None

    def test_delete_reports_whether_a_row_was_removed(self) -> None:
        removed = PostgresPlantMappingRepository(
            RecordingConnection([("DELETE FROM", [("mapping-1",)])])
        )
        absent = PostgresPlantMappingRepository(RecordingConnection())

        assert removed.delete_sync("mapping-1") is True
        assert absent.delete_sync("mapping-1") is False

    @pytest.mark.asyncio
    async def test_the_async_surface_matches_the_in_memory_implementation(self) -> None:
        """Both stores answer the same interface, so the service cannot tell them apart."""
        durable = PostgresPlantMappingRepository(PlantMappingDatabase())
        fallback = PlantMappingRepository()

        for repo in (durable, fallback):
            created = await repo.create(
                PlantMappingCreate(
                    source_plant_id="PLANT-A",
                    sister_plant_id="PLANT-B",
                    created_by="admin-user-1",
                )
            )
            assert (await repo.get_by_source_plant("PLANT-A")) != []
            assert (await repo.list_all()) != []
            assert await repo.get_by_id(created.id) is not None

            updated = await repo.update(
                created.id, PlantMappingUpdate(sister_plant_id="PLANT-C", updated_by="admin-2")
            )
            assert updated is not None and updated.sister_plant_id == "PLANT-C"
            assert await repo.update("nope", PlantMappingUpdate(updated_by="a")) is None

            assert await repo.delete(created.id) is True
            assert await repo.get_by_id(created.id) is None
            assert await repo.delete(created.id) is False

    @pytest.mark.asyncio
    async def test_the_service_raises_the_domain_error_over_the_durable_store(self) -> None:
        """The 409 rule is the service's; the durable store just supplies rows."""
        listed = [_mapping_row("mapping-1")]
        fake = RecordingConnection(
            [
                ("WHERE source_plant_id", listed),
                ("RETURNING", [_mapping_row("mapping-2")]),
            ]
        )
        service = PlantMappingService(PostgresPlantMappingRepository(fake))
        payload = PlantMappingCreate(
            source_plant_id="PLANT-A", sister_plant_id="PLANT-B", created_by="admin-user-1"
        )

        with pytest.raises(DuplicateMappingError):
            await service.create_mapping(payload)


# ---------------------------------------------------------------------------
# FR1/FR2 external accounts
# ---------------------------------------------------------------------------


class TestPostgresUserRegistry:
    def test_list_users_maps_timestamptz_to_the_dto_string(self) -> None:
        fake = RecordingConnection([("FROM admin_users", [_user_row()])])
        users = PostgresUserRegistry(fake).list_users()

        assert users == [
            UserDTO(
                user_id="u-1",
                username="dana.ops",
                email="dana.ops@example.com",
                role="3pl",
                organization="Harbor Freight",
                created_at="2026-03-01T12:00:00+00:00",
            )
        ]

    def test_list_users_orders_by_creation_time_then_username(self) -> None:
        fake = RecordingConnection(
            [
                (
                    "FROM admin_users",
                    [
                        _user_row("u-2", "zoe.ops"),
                        _user_row("u-1", "amy.ops"),
                    ],
                )
            ]
        )
        assert [u.username for u in PostgresUserRegistry(fake).list_users()] == [
            "amy.ops",
            "zoe.ops",
        ]

    def test_get_user_is_none_when_unknown(self) -> None:
        assert PostgresUserRegistry(RecordingConnection()).get_user("nope") is None

    def test_create_inserts_every_column_the_dto_carries(self) -> None:
        fake = RecordingConnection([("INSERT INTO admin_users", [_user_row()])])
        created = PostgresUserRegistry(fake).create(
            {
                "user_id": "u-1",
                "username": "dana.ops",
                "email": "dana.ops@example.com",
                "role": "3pl",
                "organization": "Harbor Freight",
                "created_at": "2026-03-01T12:00:00+00:00",
            }
        )

        sql, params = fake.statement_matching("INSERT INTO admin_users")
        assert "user_id" in sql and "organization" in sql
        assert params == (
            "u-1",
            "dana.ops",
            "dana.ops@example.com",
            "3pl",
            "Harbor Freight",
            "2026-03-01T12:00:00+00:00",
        )
        assert created.user_id == "u-1"

    def test_a_case_insensitive_duplicate_raises_before_any_insert(self) -> None:
        fake = RecordingConnection([("lower(username) = lower(%s)", [{"exists": 1}])])

        with pytest.raises(DuplicateUserError) as excinfo:
            PostgresUserRegistry(fake).create(
                {
                    "user_id": "u-2",
                    "username": "DANA.OPS",
                    "email": None,
                    "role": "3pl",
                    "organization": None,
                    "created_at": "2026-03-01T12:00:00+00:00",
                }
            )

        assert excinfo.value.username == "DANA.OPS"
        assert fake.statement_matching("INSERT INTO admin_users") is None

    def test_a_unique_violation_is_reported_as_the_typed_domain_error(self) -> None:
        class RejectingConnection(RecordingConnection):
            def respond(self, sql: str, params: Sequence[Any]) -> list[Any]:
                if _normalize(sql).startswith("INSERT INTO admin_users"):
                    raise _UniqueViolation("duplicate key value")
                return super().respond(sql, params)

        with pytest.raises(DuplicateUserError):
            PostgresUserRegistry(RejectingConnection()).create(
                {
                    "user_id": "u-2",
                    "username": "dana.ops",
                    "email": None,
                    "role": "3pl",
                    "organization": None,
                    "created_at": "2026-03-01T12:00:00+00:00",
                }
            )

    def test_update_role_returns_none_when_the_user_is_unknown(self) -> None:
        assert PostgresUserRegistry(RecordingConnection()).update_role("nope", "internal") is None

    def test_update_role_returns_the_updated_row(self) -> None:
        row = _user_row()
        row["role"] = "internal"
        fake = RecordingConnection([("UPDATE admin_users", [row])])

        updated = PostgresUserRegistry(fake).update_role("u-1", "internal")

        assert updated is not None and updated.role == "internal"
        assert fake.statement_matching("UPDATE admin_users")[1] == ("internal", "u-1")

    def test_the_module_functions_validate_before_reaching_the_store(
        self, monkeypatch
    ) -> None:
        """Roles are refused here on either backend, so the DTO shape cannot drift."""
        monkeypatch.setenv("P3_AUDIT_DSN", DSN)
        monkeypatch.setattr(
            "src.admin.users.PostgresUserRegistry",
            lambda _dsn: (_ for _ in ()).throw(AssertionError("store must not be reached")),
        )
        with pytest.raises(ValueError):
            create_external_account({"username": "dana.ops", "role": "superuser"})
        with pytest.raises(ValueError):
            update_role("u-1", "superuser")

    def test_user_not_found_surfaces_from_the_store_contract(self) -> None:
        with pytest.raises(UserNotFoundError):
            update_role("nope", "internal")


class TestJsonFallbackRemainsTheDefault:
    def test_the_documented_fallback_still_backs_every_public_function(
        self, monkeypatch, tmp_path
    ) -> None:
        monkeypatch.delenv("P3_AUDIT_DSN", raising=False)
        monkeypatch.delenv("DATABASE_URL", raising=False)
        monkeypatch.setenv("ADMIN_USERS_PATH", str(tmp_path / "admin_users.json"))

        created = create_external_account(
            {"username": "dana.ops", "email": "dana.ops@example.com", "role": "3pl"}
        )

        assert [user.username for user in list_users()] == ["dana.ops"]
        assert get_user(created.user_id) == created
        assert update_role(created.user_id, "internal").role == "internal"
        assert (tmp_path / "admin_users.json").is_file()


# ---------------------------------------------------------------------------
# FR12 audit trail
# ---------------------------------------------------------------------------


class TestPostgresAuditStorage:
    def test_append_assigns_the_chain_position_in_the_database(self) -> None:
        db = AuditTrailDatabase()
        storage = PostgresAuditStorage(db)
        service = AuditTrailService(AuditRepository(storage))

        service.record_event(AuditEventType.ACCESS, "user-1", "coa:COA-1001")
        service.record_event(AuditEventType.PRINT, "user-2", "coa:COA-1002")

        assert [row["sequence_number"] for row in db.rows] == [0, 1]
        assert db.rows[0]["event_type"] == "access"
        assert db.rows[0]["actor_id"] == "user-1"
        assert db.rows[0]["coa_id"] == "coa:COA-1001"
        assert db.rows[1]["event_type"] == "print"
        assert db.statement_matching("pg_advisory_xact_lock") is not None

    def test_append_persists_the_canonical_payload_the_hash_was_computed_over(
        self,
    ) -> None:
        db = AuditTrailDatabase()
        service = AuditTrailService(AuditRepository(PostgresAuditStorage(db)))

        entry = service.record_event(AuditEventType.ACCESS, "user-1", "coa:COA-1001")

        payload = json.loads(db.rows[0]["payload_json"])
        assert payload["canonical_payload"] == entry.compute_canonical_payload()
        assert payload["event_id"] == entry.event_id
        assert payload["details"] == entry.details

    def test_a_non_uuid_idempotency_key_still_satisfies_the_uuid_column(self) -> None:
        db = AuditTrailDatabase()
        service = AuditTrailService(AuditRepository(PostgresAuditStorage(db)))

        entry = service.record_event(
            AuditEventType.ACCESS,
            "user-1",
            "coa:COA-1001",
            idempotency_key="jde-42-COA-1001",
        )

        assert uuid.UUID(db.rows[0]["event_id"]) == uuid.uuid5(
            uuid.UUID("6f2b5f1e-0d3a-5c47-9c2e-6a1b0d4f8e31"), "jde-42-COA-1001"
        )
        assert db.rows[0]["external_event_id"] == "jde-42-COA-1001"
        assert entry.event_id == "jde-42-COA-1001"

    def test_the_chain_survives_a_round_trip_through_the_durable_store(self) -> None:
        db = AuditTrailDatabase()
        repository = AuditRepository(PostgresAuditStorage(db))
        service = AuditTrailService(repository)

        for index in range(5):
            service.record_event(AuditEventType.ACCESS, f"user-{index}", f"coa:{index}")

        reloaded = AuditTrailService(AuditRepository(PostgresAuditStorage(db)))

        assert reloaded.verify_chain_integrity() == (True, None)
        assert [entry.sequence_number for entry in reloaded.get_all_events()] == [0, 1, 2, 3, 4]
        assert reloaded.get_all_events()[-1].entry_hash == db.rows[-1]["entry_hash"]

    def test_reads_are_filtered_in_the_database_and_ordered_by_chain_position(self) -> None:
        db = AuditTrailDatabase()
        service = AuditTrailService(AuditRepository(PostgresAuditStorage(db)))
        service.record_event(AuditEventType.ACCESS, "user-1", "coa:COA-1001")
        service.record_event(AuditEventType.PRINT, "user-2", "coa:COA-1002")
        service.record_event(AuditEventType.ACCESS, "user-3", "coa:COA-1003")
        storage = PostgresAuditStorage(db)

        assert [e.resource_id for e in storage.get_by_resource("coa:COA-1001")] == [
            "coa:COA-1001"
        ]
        assert [e.resource_id for e in storage.get_by_type(AuditEventType.ACCESS)] == [
            "coa:COA-1001",
            "coa:COA-1003",
        ]
        assert storage.count() == 3
        cutoff = datetime(2030, 1, 1, tzinfo=UTC)
        assert len(storage.get_older_than(cutoff)) == 3
        assert db.statement_matching("WHERE coa_id = %s") is not None
        assert db.statement_matching("WHERE event_type = %s") is not None
        assert db.statement_matching("WHERE occurred_at < %s") is not None

    def test_get_all_asks_for_the_chain_ordered_read(self) -> None:
        db = AuditTrailDatabase()
        PostgresAuditStorage(db).get_all()
        sql, _ = db.statement_matching("FROM audit_events")
        assert sql.endswith("ORDER BY sequence_number")

    def test_no_update_or_delete_path_exists_on_the_durable_store(self) -> None:
        for name in ("update", "delete", "remove", "purge"):
            assert not hasattr(PostgresAuditStorage, name)

    def test_the_audit_service_writes_through_the_selected_backend(self, monkeypatch) -> None:
        """Selecting the durable store must not require a reachable server."""
        monkeypatch.setenv("P3_AUDIT_DSN", DSN)
        monkeypatch.delenv("DATABASE_URL", raising=False)
        repository = AuditRepository()
        assert isinstance(repository._storage, PostgresAuditStorage)

    def test_the_default_audit_backend_stays_in_memory_without_a_dsn(
        self, monkeypatch
    ) -> None:
        monkeypatch.delenv("P3_AUDIT_DSN", raising=False)
        monkeypatch.delenv("DATABASE_URL", raising=False)
        assert isinstance(AuditRepository()._storage, InMemoryAuditStorage)


# ---------------------------------------------------------------------------
# The migration really declares what the repositories read
# ---------------------------------------------------------------------------


class TestSchemaCoherence:
    """Every column the repositories name must exist in the checked-in migration.

    A canned-response test cannot catch SQL that is internally consistent but
    names a column the migration never creates - the failure only appears at
    runtime, against a migrated database. This reads the migration instead.
    """

    @pytest.mark.parametrize(
        "column",
        [
            "mapping_uuid",
            "source_plant_id",
            "target_plant_id",
            "active",
            "created_by",
            "updated_by",
            "created_at",
            "updated_at",
        ],
    )
    def test_plant_mapping_columns_are_declared(self, migration: str, column: str) -> None:
        assert re.search(rf"\b{column}\b", migration), f"plant_mappings.{column} is missing"

    @pytest.mark.parametrize(
        "column",
        ["user_id", "username", "email", "role", "organization", "created_at", "updated_at"],
    )
    def test_admin_user_columns_are_declared(self, migration: str, column: str) -> None:
        assert re.search(r"\badmin_users\b", migration)
        assert re.search(rf"\b{column}\b", migration), f"admin_users.{column} is missing"

    @pytest.mark.parametrize(
        "column",
        [
            "sequence",
            "event_id",
            "external_event_id",
            "event_type",
            "actor_id",
            "coa_id",
            "occurred_at",
            "payload_json",
            "prev_hash",
            "entry_hash",
            "sequence_number",
            "details",
            "ip_address",
        ],
    )
    def test_audit_event_columns_are_declared(self, migration: str, column: str) -> None:
        assert re.search(rf"\b{column}\b", migration), f"audit_events.{column} is missing"

    @pytest.mark.parametrize(
        "table", ["coa_metadata", "audit_events", "plant_mappings", "admin_users"]
    )
    def test_every_table_is_created_idempotently(self, migration: str, table: str) -> None:
        assert f"CREATE TABLE IF NOT EXISTS {table}" in migration

    def test_the_extensions_are_idempotent_too(self, migration: str) -> None:
        added = re.findall(r"ALTER TABLE (\w+) ADD COLUMN IF NOT EXISTS", migration)
        assert "plant_mappings" in added
        assert added.count("audit_events") >= 4

    def test_the_migration_wrapped_in_one_transaction(self, migration: str) -> None:
        assert migration.index("BEGIN;") < migration.index("COMMIT;")

    def test_the_migration_names_the_service_that_applies_it(self, migration: str) -> None:
        assert "migrate" in migration
        assert "ON_ERROR_STOP" in migration


# ---------------------------------------------------------------------------
# A real server, when one is reachable
# ---------------------------------------------------------------------------


def _live_dsn() -> str:
    """Return a reachable audit DSN, or skip the test."""
    dsn = os.getenv("P3_TEST_AUDIT_DSN") or os.getenv("P3_AUDIT_DSN")
    if not dsn:
        pytest.skip("no live audit DSN configured (set P3_TEST_AUDIT_DSN)")
    psycopg2 = pytest.importorskip("psycopg2")
    try:
        conn = psycopg2.connect(dsn, connect_timeout=3)
    except Exception as exc:  # noqa: BLE001 - any driver error means the live DB is unavailable
        pytest.skip(f"live audit database is unavailable: {exc}")
    with conn, conn.cursor() as cursor:
        cursor.execute(
            "SELECT count(*) FROM information_schema.tables "
            "WHERE table_schema = 'public' AND table_name IN "
            "('audit_events', 'plant_mappings', 'admin_users')"
        )
        row = cursor.fetchone()
    if not row or int(row[0]) < 3:
        pytest.skip("live audit database has not had migrations applied")
    return dsn


class TestLiveRoundTrip:
    """The same repositories, against a real server, when one is available."""

    def test_an_account_round_trips_through_postgres(self, monkeypatch) -> None:
        dsn = _live_dsn()
        monkeypatch.setenv("P3_AUDIT_DSN", dsn)
        monkeypatch.setenv("ADMIN_USERS_PATH", "/tmp/never-written.json")
        unique = f"live.agent.{uuid.uuid4().hex[:8]}"

        created = create_external_account(
            {"username": unique, "email": "live@example.com", "role": "internal"}
        )
        try:
            assert get_user(created.user_id) == created
            assert unique in [user.username for user in list_users()]
            assert update_role(created.user_id, "customer").role == "customer"
        finally:
            with connection(dsn) as conn, conn.cursor() as cursor:
                cursor.execute("DELETE FROM admin_users WHERE user_id = %s", (created.user_id,))

    @pytest.mark.asyncio
    async def test_a_mapping_round_trips_through_postgres(self) -> None:
        dsn = _live_dsn()
        repo = PostgresPlantMappingRepository(dsn)

        created = await repo.create(
            PlantMappingCreate(
                source_plant_id="LIVE-A",
                sister_plant_id="LIVE-B",
                created_by="live-agent",
            )
        )
        try:
            assert (await repo.get_by_id(created.id)).sister_plant_id == "LIVE-B"
            assert [m.id for m in await repo.get_by_source_plant("LIVE-A")] == [created.id]
            assert await repo.delete(created.id) is True
            assert await repo.get_by_id(created.id) is None
        finally:
            with connection(dsn) as conn, conn.cursor() as cursor:
                cursor.execute(
                    "DELETE FROM plant_mappings WHERE mapping_uuid = %s", (created.id,)
                )

    def test_an_audit_chain_verifies_after_a_round_trip(self) -> None:
        dsn = _live_dsn()
        storage = PostgresAuditStorage(dsn)
        service = AuditTrailService(AuditRepository(storage))
        tag = f"LIVE-{uuid.uuid4().hex[:8]}"

        before = storage.count()
        first = service.record_event(AuditEventType.ACCESS, "live-agent", f"coa:{tag}-1")
        second = service.record_event(AuditEventType.ACCESS, "live-agent", f"coa:{tag}-2")

        assert second.sequence_number == before + 1
        assert AuditTrailService(AuditRepository(PostgresAuditStorage(dsn))).verify_chain_integrity() == (
            True,
            None,
        )
        assert [entry.event_id for entry in storage.get_by_resource(f"coa:{tag}-1")] == [
            first.event_id
        ]
        assert storage.get_by_resource(f"coa:{tag}-2")[0].entry_hash == second.entry_hash
        assert storage.get_older_than(datetime(2030, 1, 1, tzinfo=UTC))

        with connection(dsn) as conn, conn.cursor() as cursor:
            cursor.execute("DELETE FROM audit_events WHERE coa_id LIKE %s", (f"coa:{tag}-%",))
