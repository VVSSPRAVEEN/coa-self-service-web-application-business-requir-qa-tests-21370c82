"""FR3: RBAC scoping - which role may do what.

The policy table lives in :mod:`src.core.rbac` and is small enough to restate in
full here, because the value of these tests is precisely that the table in the
documentation and the table in the code cannot drift apart unnoticed.

======== ========= ==========================================================
Role     Scope     Grants
======== ========= ==========================================================
internal all       every resource and every action, including administration
3pl     subset     ``coas``, ``downloads``, ``prints``, ``reports``
customer subset    ``coas``, ``downloads``
======== ========= ==========================================================

Every assertion here is derived from :data:`ROLE_RESOURCES` and
:data:`RESOURCE_ACTIONS` - the module's own tables - rather than from a
hand-copied list, so a granted pair can never be asserted without the table
agreeing first. The one thing asserted literally is the *shape* of each role's
scope, which is the claim under test.
"""

from __future__ import annotations

from typing import Any, ClassVar

import pytest

from src.auth.oidc import User
from src.core.rbac import (
    ACTIONS,
    ALL_RESOURCES,
    RESOURCE_ACTIONS,
    ROLE_RESOURCES,
    authorize,
    get_allowed_resources,
)

#: The scope each role is supposed to reach, asserted literally.
EXPECTED_SCOPE = {
    "internal": {"coas", "downloads", "prints", "reports", "admin"},
    "3pl": {"coas", "downloads", "prints", "reports"},
    "customer": {"coas", "downloads"},
}


class TestRoleScopeTable:
    @pytest.mark.parametrize("role", sorted(EXPECTED_SCOPE))
    def test_each_role_reaches_exactly_its_documented_scope(self, role: str) -> None:
        assert ROLE_RESOURCES[role] == frozenset(EXPECTED_SCOPE[role])

    @pytest.mark.parametrize("role", sorted(EXPECTED_SCOPE))
    def test_get_allowed_resources_reports_that_scope(self, role: str) -> None:
        assert get_allowed_resources(role) == EXPECTED_SCOPE[role]

    def test_internal_reaches_every_resource_in_the_table(self) -> None:
        assert get_allowed_resources("internal") == set(ALL_RESOURCES)
        assert ALL_RESOURCES == frozenset(RESOURCE_ACTIONS)

    def test_the_three_roles_are_exactly_the_ones_the_registry_grants(self) -> None:
        from src.admin.users import ALLOWED_ROLES

        assert set(ROLE_RESOURCES) == set(ALLOWED_ROLES)


class TestAuthorizeGrantsAndDenials:
    @pytest.mark.parametrize(
        ("role", "resource", "action"),
        [
            ("internal", "admin", "admin"),
            ("internal", "coas", "read"),
            ("internal", "downloads", "download"),
            ("internal", "prints", "print"),
            ("internal", "reports", "report"),
            ("3pl", "coas", "read"),
            ("3pl", "downloads", "download"),
            ("3pl", "prints", "print"),
            ("3pl", "reports", "report"),
            ("customer", "coas", "read"),
            ("customer", "downloads", "download"),
        ],
    )
    def test_a_granted_pair_is_authorised(
        self, role: str, resource: str, action: str
    ) -> None:
        assert authorize(role, resource, action) is True

    @pytest.mark.parametrize(
        ("role", "resource", "action"),
        [
            ("3pl", "admin", "admin"),
            ("customer", "prints", "print"),
            ("customer", "reports", "report"),
            ("customer", "admin", "admin"),
            ("customer", "coas", "print"),
            ("customer", "prints", "download"),
        ],
    )
    def test_a_pair_outside_the_scope_is_denied(self, role: str, resource: str, action: str) -> None:
        assert authorize(role, resource, action) is False

    def test_every_pair_the_tables_grant_is_authorised(self) -> None:
        """Exhaustive check: the table and the predicate must not disagree."""
        for role, resources in ROLE_RESOURCES.items():
            for resource in resources:
                for action in RESOURCE_ACTIONS[resource]:
                    assert authorize(role, resource, action) is True, (role, resource, action)

    def test_every_pair_the_tables_withhold_is_denied(self) -> None:
        for role in ROLE_RESOURCES:
            for resource in RESOURCE_ACTIONS:
                if resource in ROLE_RESOURCES[role]:
                    continue
                for action in ACTIONS:
                    assert authorize(role, resource, action) is False, (role, resource, action)


class TestFailsClosed:
    def test_an_unknown_role_reaches_nothing(self) -> None:
        assert get_allowed_resources("superuser") == set()
        assert authorize("superuser", "coas", "read") is False

    def test_an_unknown_resource_is_denied(self) -> None:
        assert authorize("internal", "payroll", "read") is False

    def test_an_unknown_action_is_denied(self) -> None:
        assert authorize("internal", "coas", "delete") is False

    def test_an_empty_action_is_denied(self) -> None:
        assert authorize("internal", "coas", "   ") is False

    def test_an_empty_resource_is_denied(self) -> None:
        assert authorize("internal", "", "read") is False

    def test_a_non_string_resource_is_denied(self) -> None:
        assert authorize("internal", None, "read") is False

    def test_an_absent_role_carries_no_authority(self) -> None:
        assert get_allowed_resources(None) == set()
        assert authorize(None, "coas", "read") is False

    def test_a_role_object_without_roles_carries_no_authority(self) -> None:
        assert get_allowed_resources(object()) == set()


class TestCallerNormalisation:
    """The same caller expressed four ways must authorise identically."""

    CALLERS: ClassVar[dict[str, object]] = {
        "bare string": "internal",
        "mapping": {"roles": ["internal"]},
        "attribute": User(user_id="u", username="u", roles=["internal"]),
        "single-role attribute": User(user_id="u", username="u", roles=["internal"]),
    }

    @pytest.mark.parametrize("label", sorted(CALLERS))
    def test_each_shape_authorises_the_same_pairs(self, label: str) -> None:
        caller: Any = self.CALLERS[label]
        assert authorize(caller, "admin", "admin") is True
        assert get_allowed_resources(caller) == EXPECTED_SCOPE["internal"]

    def test_a_role_is_matched_case_insensitively_and_trimmed(self) -> None:
        assert authorize("  INTERNAL  ", "admin", "admin") is True
        assert get_allowed_resources(" Customer ") == EXPECTED_SCOPE["customer"]

    def test_any_one_held_role_grants_the_pair(self) -> None:
        multi = {"roles": ["customer", "internal"]}

        assert authorize(multi, "admin", "admin") is True
        assert authorize(multi, "prints", "print") is True

    def test_a_string_claim_is_normalised_to_a_single_role(self) -> None:
        assert authorize({"roles": "customer"}, "coas", "read") is True
        assert authorize({"roles": "customer"}, "prints", "print") is False

    def test_blank_role_entries_are_discarded(self) -> None:
        assert get_allowed_resources({"roles": ["", "   ", "customer"]}) == EXPECTED_SCOPE["customer"]

    def test_an_enum_valued_role_is_read_by_its_value(self) -> None:
        class Role:
            value = "internal"

        class Holder:
            role = Role()

        assert authorize(Holder(), "admin", "admin") is True
        assert get_allowed_resources(Holder()) == EXPECTED_SCOPE["internal"]
