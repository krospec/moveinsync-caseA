"""DEMO token issuer - a stand-in for the client company's IdP.

This endpoint exists so the case study can be run and evaluated end to end.
In production, tokens would be minted by each client company's identity
provider; the rest of the service would not change, because it only ever
reads claims from a verified JWT.
"""
from fastapi import APIRouter, Depends, HTTPException, status

from ..auth import create_token
from ..config import get_settings
from ..db import SessionLocal
from ..models import Employee
from ..schemas import TokenIn, TokenOut

router = APIRouter(tags=["auth"])


@router.post("/auth/token", response_model=TokenOut)
async def issue_demo_token(body: TokenIn, settings=Depends(get_settings)) -> TokenOut:
    async with SessionLocal() as session:  # type: AsyncSession
        employee = await session.get(Employee, body.employee_id)
        if employee is None:
            raise HTTPException(status.HTTP_404_NOT_FOUND, "Unknown employee id.")
        return TokenOut(
            access_token=create_token(employee, settings),
            employee_id=employee.id,
            company_id=employee.company_id,
            role=employee.role,
            name=employee.name,
        )
