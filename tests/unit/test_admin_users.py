"""Unit tests for the administrator-managed external account registry.

Every test repoints ``ADMIN_USERS_PATH`` at a temporary file, so the suite never
reads or writes the deployment's ``./data/admin_users.json`` and never needs
cleanup between runs.
"""

from __future__ import annotations

import json
from datetime import datetime

import pytest

from src.admin.users import (
    ALLOWED_ROLES,
    DEFAULT_STORE_PATH,
    DuplicateUserError,
    InvalidRoleError,
    UserDTO,
    UserNotFoundError,
    create_external_account,
    get_user,
    list_users,
    update_role,
)


@pytest.fixture(autouse=True)
def registry_path(tmp_path, monkeypatch):
    """Point the registry at an empty file inside the test's tmp directory."""
    path = tmp_path / "state" / "admin_users.json"
    monkeypatch.setenv("ADMIN_USERS_PATH", str(path))
    return path


def _payload(**overrides):
    base = {
        "username": "dana.ops",
        "email": "dana.ops@example.com",
        "role": "3pl",
        "organization": "Harbor Freight",
    }
    base.update(overrides)
    return base


class TestCreateExternalAccount:
    def test_creates_and_persists_an_account(self, registry_path) -> None:
        user = create_external_account(_payload())

        assert isinstance(user, UserDTO)
        assert user.user_id
        assert user.username == "dana.ops"
        assert user.email == "dana.ops@example.com"
        assert user.role == "3pl"
        assert user.organization == "Harbor Freight"
        assert datetime.fromisoformat(user.created_at).tzinfo is not None

        stored = json.loads(registry_path.read_text(encoding="utf-8"))
        assert [row["username"] for row in stored] == ["dana.ops"]

    def test_accepts_every_allowed_role(self) -> None:
        for index, role in enumerate(ALLOWED_ROLES):
            user = create_external_account(_payload(username=f"user{index}", role=role))
            assert user.role == role

    def test_rejects_a_duplicate_username(self) -> None:
        create_external_account(_payload())

        with pytest.raises(DuplicateUserError) as excinfo:
            create_external_account(_payload(email="other@example.com"))

        assert excinfo.value.username == "dana.ops"
        assert len(list_users()) == 1

    def test_duplicate_detection_ignores_case(self) -> None:
        create_external_account(_payload())

        with pytest.raises(DuplicateUserError):
            create_external_account(_payload(username="DANA.OPS"))

    def test_rejects_an_unknown_role(self) -> None:
        with pytest.raises(InvalidRoleError) as excinfo:
            create_external_account(_payload(role="superuser"))

        assert excinfo.value.role == "superuser"
        # InvalidRoleError is a ValueError, so generic callers still catch it.
        with pytest.raises(ValueError):
            create_external_account(_payload(role="superuser"))
        assert list_users() == []

    def test_rejects_an_empty_role(self) -> None:
        with pytest.raises(InvalidRoleError):
            create_external_account(_payload(role="   "))

    def test_rejects_a_malformed_email(self) -> None:
        with pytest.raises(ValueError):
            create_external_account(_payload(email="not-an-address"))

    def test_rejects_an_empty_username(self) -> None:
        with pytest.raises(ValueError):
            create_external_account(_payload(username="  "))

    def test_email_and_organization_are_optional(self) -> None:
        user = create_external_account(
            {"username": "minimal.user", "role": "customer"}
        )

        assert user.email is None
        assert user.organization is None


class TestListUsers:
    def test_empty_registry_lists_nothing(self) -> None:
        assert list_users() == []

    def test_lists_persisted_accounts_in_creation_order(self) -> None:
        first = create_external_account(_payload(username="a.user"))
        second = create_external_account(_payload(username="b.user", role="customer"))

        users = list_users()

        assert [user.user_id for user in users] == [first.user_id, second.user_id]

    def test_survives_a_corrupt_store_document(self, registry_path) -> None:
        registry_path.parent.mkdir(parents=True, exist_ok=True)
        registry_path.write_text("{not json", encoding="utf-8")

        assert list_users() == []

    def test_reads_back_what_a_previous_call_wrote(self) -> None:
        created = create_external_account(_payload())

        assert get_user(created.user_id) == created
        assert get_user("unknown-id") is None


class TestUpdateRole:
    def test_updates_and_persists_the_role(self, registry_path) -> None:
        created = create_external_account(_payload())

        updated = update_role(created.user_id, "internal")

        assert updated.role == "internal"
        assert updated.user_id == created.user_id
        stored = json.loads(registry_path.read_text(encoding="utf-8"))
        assert stored[0]["role"] == "internal"
        assert get_user(created.user_id).role == "internal"

    def test_rejects_an_unknown_role(self) -> None:
        created = create_external_account(_payload())

        with pytest.raises(InvalidRoleError):
            update_role(created.user_id, "root")

        assert get_user(created.user_id).role == "3pl"

    def test_reports_an_unknown_user(self) -> None:
        with pytest.raises(UserNotFoundError) as excinfo:
            update_role("does-not-exist", "internal")

        assert excinfo.value.user_id == "does-not-exist"


def test_default_store_path_is_resolved_under_the_working_directory(
    monkeypatch, tmp_path
) -> None:
    monkeypatch.delenv("ADMIN_USERS_PATH", raising=False)
    monkeypatch.chdir(tmp_path)

    created = create_external_account(_payload())

    assert DEFAULT_STORE_PATH == "./data/admin_users.json"
    assert (tmp_path / "data" / "admin_users.json").is_file()
    assert created.username == "dana.ops"
