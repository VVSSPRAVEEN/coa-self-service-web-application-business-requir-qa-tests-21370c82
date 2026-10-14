"""Unit tests for the FR9 sister-plant mapping domain and its HTTP surface.

The mapping row, the create/update payloads and the repository are the single
definitions owned by ``src.admin.plant_mapping.models`` and
``src.admin.plant_mapping.repository``; this module imports those same objects,
so a test can never drift onto a private copy of the model.

The service raises the typed errors :class:`DuplicateMappingError` and
:class:`MappingNotFoundError`; translating them into 409/404 is the router's
job, so both layers are covered here: the service tests assert the typed
errors, and the HTTP tests assert the status codes a client actually sees.
"""

from __future__ import annotations

from typing import Iterator

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.admin.plant_mapping.models import (
    PlantMappingCreate,
    PlantMappingUpdate,
)
from src.admin.plant_mapping.repository import PlantMappingRepository
from src.admin.plant_mapping.router import (
    get_plant_mapping_service,
    register_plant_mappings,
)
from src.admin.plant_mapping.service import (
    DuplicateMappingError,
    MappingNotFoundError,
    PlantMappingService,
)


@pytest.fixture
def repo() -> PlantMappingRepository:
    return PlantMappingRepository()


@pytest.fixture
def service(repo: PlantMappingRepository) -> PlantMappingService:
    return PlantMappingService(repository=repo)


@pytest.fixture
def client(repo: PlantMappingRepository) -> Iterator[TestClient]:
    """A TestClient over the real router, backed by an isolated repository."""
    application = FastAPI()
    register_plant_mappings(application)
    isolated_service = PlantMappingService(repository=repo)
    application.dependency_overrides[get_plant_mapping_service] = (
        lambda: isolated_service
    )
    with TestClient(application) as test_client:
        yield test_client


@pytest.mark.asyncio
async def test_create_mapping_success(service: PlantMappingService) -> None:
    payload = PlantMappingCreate(
        source_plant_id="PLANT-A",
        sister_plant_id="PLANT-B",
        created_by="admin-user-1",
    )
    result = await service.create_mapping(payload)
    assert result.source_plant_id == "PLANT-A"
    assert result.sister_plant_id == "PLANT-B"
    assert result.is_active is True
    assert result.created_by == "admin-user-1"
    assert result.id is not None


@pytest.mark.asyncio
async def test_create_duplicate_active_mapping_raises_conflict(
    service: PlantMappingService,
) -> None:
    payload = PlantMappingCreate(
        source_plant_id="PLANT-A",
        sister_plant_id="PLANT-B",
        created_by="admin-user-1",
    )
    await service.create_mapping(payload)
    with pytest.raises(DuplicateMappingError):
        await service.create_mapping(payload)


@pytest.mark.asyncio
async def test_get_mapping_not_found(service: PlantMappingService) -> None:
    with pytest.raises(MappingNotFoundError):
        await service.get_mapping("nonexistent-id")


@pytest.mark.asyncio
async def test_get_mapping_success(service: PlantMappingService) -> None:
    payload = PlantMappingCreate(
        source_plant_id="PLANT-A",
        sister_plant_id="PLANT-B",
        created_by="admin-user-1",
    )
    created = await service.create_mapping(payload)
    fetched = await service.get_mapping(created.id)
    assert fetched.id == created.id
    assert fetched.source_plant_id == "PLANT-A"


@pytest.mark.asyncio
async def test_list_mappings(service: PlantMappingService) -> None:
    await service.create_mapping(
        PlantMappingCreate(
            source_plant_id="PLANT-A",
            sister_plant_id="PLANT-B",
            created_by="admin-user-1",
        )
    )
    await service.create_mapping(
        PlantMappingCreate(
            source_plant_id="PLANT-A",
            sister_plant_id="PLANT-C",
            created_by="admin-user-1",
        )
    )
    mappings = await service.list_mappings()
    assert len(mappings) == 2


@pytest.mark.asyncio
async def test_list_by_source_plant(service: PlantMappingService) -> None:
    await service.create_mapping(
        PlantMappingCreate(
            source_plant_id="PLANT-A",
            sister_plant_id="PLANT-B",
            created_by="admin-user-1",
        )
    )
    await service.create_mapping(
        PlantMappingCreate(
            source_plant_id="PLANT-X",
            sister_plant_id="PLANT-Y",
            created_by="admin-user-1",
        )
    )
    results = await service.list_by_source_plant("PLANT-A")
    assert len(results) == 1
    assert results[0].source_plant_id == "PLANT-A"


@pytest.mark.asyncio
async def test_update_mapping(service: PlantMappingService) -> None:
    created = await service.create_mapping(
        PlantMappingCreate(
            source_plant_id="PLANT-A",
            sister_plant_id="PLANT-B",
            created_by="admin-user-1",
        )
    )
    update_payload = PlantMappingUpdate(
        sister_plant_id="PLANT-C",
        updated_by="admin-user-2",
    )
    updated = await service.update_mapping(created.id, update_payload)
    assert updated is not None
    assert updated.sister_plant_id == "PLANT-C"
    assert updated.updated_by == "admin-user-2"


@pytest.mark.asyncio
async def test_update_mapping_deactivate(service: PlantMappingService) -> None:
    created = await service.create_mapping(
        PlantMappingCreate(
            source_plant_id="PLANT-A",
            sister_plant_id="PLANT-B",
            created_by="admin-user-1",
        )
    )
    update_payload = PlantMappingUpdate(
        is_active=False,
        updated_by="admin-user-1",
    )
    updated = await service.update_mapping(created.id, update_payload)
    assert updated is not None
    assert updated.is_active is False


@pytest.mark.asyncio
async def test_update_mapping_not_found(service: PlantMappingService) -> None:
    with pytest.raises(MappingNotFoundError):
        await service.update_mapping(
            "nonexistent",
            PlantMappingUpdate(updated_by="admin-user-1"),
        )


@pytest.mark.asyncio
async def test_delete_mapping(service: PlantMappingService) -> None:
    created = await service.create_mapping(
        PlantMappingCreate(
            source_plant_id="PLANT-A",
            sister_plant_id="PLANT-B",
            created_by="admin-user-1",
        )
    )
    await service.delete_mapping(created.id)
    with pytest.raises(MappingNotFoundError):
        await service.get_mapping(created.id)


@pytest.mark.asyncio
async def test_delete_mapping_not_found(service: PlantMappingService) -> None:
    with pytest.raises(MappingNotFoundError):
        await service.delete_mapping("nonexistent")


@pytest.mark.asyncio
async def test_get_active_sister_plants(service: PlantMappingService) -> None:
    await service.create_mapping(
        PlantMappingCreate(
            source_plant_id="PLANT-A",
            sister_plant_id="PLANT-B",
            created_by="admin-user-1",
        )
    )
    created_c = await service.create_mapping(
        PlantMappingCreate(
            source_plant_id="PLANT-A",
            sister_plant_id="PLANT-C",
            created_by="admin-user-1",
        )
    )
    await service.update_mapping(
        created_c.id,
        PlantMappingUpdate(is_active=False, updated_by="admin-user-1"),
    )
    active = await service.get_active_sister_plants("PLANT-A")
    assert active == ["PLANT-B"]


def test_update_payload_carries_the_field_the_repository_assigns() -> None:
    """The PATCH payload and ``PlantMappingRepository.update`` must agree.

    ``repository.update`` assigns ``existing.updated_by = payload.updated_by``,
    so the model the router validates against has to declare ``updated_by``;
    otherwise the assignment raises ``AttributeError`` and the route 500s.
    """
    field = PlantMappingUpdate.model_fields["updated_by"]
    assert field.is_required()
    assert PlantMappingUpdate(updated_by="admin-user-1").updated_by == "admin-user-1"


def test_patch_round_trip_is_never_a_server_error(client: TestClient) -> None:
    """create -> PATCH -> list, the reviewer's reported 500 path, end to end.

    Regression guard for the defect where the router validated a payload model
    without ``updated_by`` while the repository dereferenced it: the request
    reached the handler and blew up with a 500 instead of returning the mapping.
    """
    created = client.post(
        "/admin/plant-mappings",
        json={
            "source_plant_id": "PLANT-A",
            "sister_plant_id": "PLANT-B",
            "created_by": "admin-user-1",
        },
    )
    assert created.status_code == 201
    mapping_id = created.json()["id"]

    patched = client.patch(
        f"/admin/plant-mappings/{mapping_id}",
        json={"sister_plant_id": "PLANT-C", "updated_by": "admin-user-2"},
    )
    assert patched.status_code == 200
    assert patched.json()["sister_plant_id"] == "PLANT-C"
    assert patched.json()["updated_by"] == "admin-user-2"

    listed = client.get("/admin/plant-mappings")
    assert listed.status_code == 200
    assert [row["id"] for row in listed.json()] == [mapping_id]

    # A payload that omits the audited actor is rejected by validation (422),
    # never by an unhandled attribute access (500).
    missing_actor = client.patch(
        f"/admin/plant-mappings/{mapping_id}", json={"is_active": False}
    )
    assert missing_actor.status_code == 422


def test_patch_unknown_mapping_is_404(client: TestClient) -> None:
    response = client.patch(
        "/admin/plant-mappings/nonexistent",
        json={"is_active": False, "updated_by": "admin-user-1"},
    )
    assert response.status_code == 404


def test_duplicate_create_is_409(client: TestClient) -> None:
    body = {
        "source_plant_id": "PLANT-A",
        "sister_plant_id": "PLANT-B",
        "created_by": "admin-user-1",
    }
    assert client.post("/admin/plant-mappings", json=body).status_code == 201
    assert client.post("/admin/plant-mappings", json=body).status_code == 409


def test_unknown_mapping_is_404(client: TestClient) -> None:
    assert client.get("/admin/plant-mappings/nonexistent").status_code == 404
    assert client.delete("/admin/plant-mappings/nonexistent").status_code == 404