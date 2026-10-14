from __future__ import annotations

import pytest
from fastapi import HTTPException

from src.validation.sister_plant.models import (
    PrintValidationStatus,
    SisterPlantPrintRequest,
)
from src.validation.sister_plant.service import SisterPlantPrintService
from src.validation.sister_plant.validator import SisterPlantValidator


def _mock_sister_plants(source_plant_id: str) -> list[str]:
    mapping = {
        "PLANT-A": ["PLANT-B", "PLANT-C"],
        "PLANT-B": ["PLANT-A"],
    }
    return mapping.get(source_plant_id, [])


@pytest.fixture
def validator() -> SisterPlantValidator:
    return SisterPlantValidator(get_sister_plants_fn=_mock_sister_plants)


@pytest.fixture
def service(validator: SisterPlantValidator) -> SisterPlantPrintService:
    return SisterPlantPrintService(validator=validator)


def test_same_plant_print_is_allowed(validator: SisterPlantValidator) -> None:
    request = SisterPlantPrintRequest(
        coa_id="COA-001",
        requesting_plant_id="PLANT-A",
        target_plant_id="PLANT-A",
        user_id="user-1",
    )
    result = validator.validate_print_request(request)
    assert result.is_valid is True
    assert result.status == PrintValidationStatus.ALLOWED


def test_valid_sister_plant_print_is_allowed(
    validator: SisterPlantValidator,
) -> None:
    request = SisterPlantPrintRequest(
        coa_id="COA-002",
        requesting_plant_id="PLANT-A",
        target_plant_id="PLANT-B",
        user_id="user-1",
    )
    result = validator.validate_print_request(request)
    assert result.is_valid is True
    assert result.status == PrintValidationStatus.ALLOWED
    assert "valid sister-plant mapping" in result.message


def test_invalid_sister_plant_print_is_denied(
    validator: SisterPlantValidator,
) -> None:
    request = SisterPlantPrintRequest(
        coa_id="COA-003",
        requesting_plant_id="PLANT-A",
        target_plant_id="PLANT-Z",
        user_id="user-1",
    )
    result = validator.validate_print_request(request)
    assert result.is_valid is False
    assert result.status == PrintValidationStatus.DENIED
    assert "no active mapping" in result.message


def test_bidirectional_mapping_respected(
    validator: SisterPlantValidator,
) -> None:
    request = SisterPlantPrintRequest(
        coa_id="COA-004",
        requesting_plant_id="PLANT-B",
        target_plant_id="PLANT-A",
        user_id="user-2",
    )
    result = validator.validate_print_request(request)
    assert result.is_valid is True
    assert result.status == PrintValidationStatus.ALLOWED


def test_unmapped_source_plant_denied(
    validator: SisterPlantValidator,
) -> None:
    request = SisterPlantPrintRequest(
        coa_id="COA-005",
        requesting_plant_id="PLANT-X",
        target_plant_id="PLANT-A",
        user_id="user-3",
    )
    result = validator.validate_print_request(request)
    assert result.is_valid is False
    assert result.status == PrintValidationStatus.DENIED


def test_audit_log_records_every_validation(
    validator: SisterPlantValidator,
) -> None:
    req1 = SisterPlantPrintRequest(
        coa_id="COA-010",
        requesting_plant_id="PLANT-A",
        target_plant_id="PLANT-B",
        user_id="user-1",
    )
    req2 = SisterPlantPrintRequest(
        coa_id="COA-011",
        requesting_plant_id="PLANT-A",
        target_plant_id="PLANT-Z",
        user_id="user-1",
        reason="Testing denied path",
    )
    validator.validate_print_request(req1)
    validator.validate_print_request(req2)

    audit = validator.get_audit_log()
    assert len(audit) == 2
    assert audit[0].coa_id == "COA-010"
    assert audit[0].validation_status == PrintValidationStatus.ALLOWED
    assert audit[1].coa_id == "COA-011"
    assert audit[1].validation_status == PrintValidationStatus.DENIED
    assert audit[1].reason == "Testing denied path"


def test_clear_audit_log(validator: SisterPlantValidator) -> None:
    req = SisterPlantPrintRequest(
        coa_id="COA-020",
        requesting_plant_id="PLANT-A",
        target_plant_id="PLANT-B",
        user_id="user-1",
    )
    validator.validate_print_request(req)
    assert len(validator.get_audit_log()) == 1
    validator.clear_audit_log()
    assert len(validator.get_audit_log()) == 0


@pytest.mark.asyncio
async def test_service_validate_and_audit_allowed(
    service: SisterPlantPrintService,
) -> None:
    request = SisterPlantPrintRequest(
        coa_id="COA-100",
        requesting_plant_id="PLANT-A",
        target_plant_id="PLANT-B",
        user_id="user-1",
    )
    result = await service.validate_and_audit(request)
    assert result.is_valid is True
    assert result.status == PrintValidationStatus.ALLOWED


@pytest.mark.asyncio
async def test_service_validate_and_audit_denied_raises_403(
    service: SisterPlantPrintService,
) -> None:
    request = SisterPlantPrintRequest(
        coa_id="COA-101",
        requesting_plant_id="PLANT-A",
        target_plant_id="PLANT-Z",
        user_id="user-1",
    )
    with pytest.raises(HTTPException) as exc_info:
        await service.validate_and_audit(request)
    assert exc_info.value.status_code == 403
    assert "no active mapping" in exc_info.value.detail


@pytest.mark.asyncio
async def test_service_audit_log_accessible(
    service: SisterPlantPrintService,
) -> None:
    request = SisterPlantPrintRequest(
        coa_id="COA-200",
        requesting_plant_id="PLANT-A",
        target_plant_id="PLANT-B",
        user_id="user-1",
    )
    await service.validate_and_audit(request)
    audit = await service.get_audit_log()
    assert len(audit) == 1
    assert audit[0].coa_id == "COA-200"
