"""JWT issuing and verification.

Identity (employee, company, role) comes ONLY from the token. The chat API has
no field anywhere that could carry identity from a request body or from the
LLM. /auth/token is a demo stand-in for the client company's own IdP: in the
real platform, MoveInSync would validate tokens minted by each company's
identity provider, with the same claims.
"""
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

import jwt as pyjwt
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from .config import Settings, get_settings
from .models import Employee


@dataclass(frozen=True)
class Identity:
    employee_id: str
    company_id: str
    role: str
    name: str


def create_token(employee: Employee, settings: Settings | None = None) -> str:
    settings = settings or get_settings()
    now = datetime.now(timezone.utc)
    payload = {
        "sub": employee.id,
        "company_id": employee.company_id,
        "role": employee.role,
        "name": employee.name,
        "iat": int(now.timestamp()),
        "exp": int((now + timedelta(minutes=settings.jwt_ttl_minutes)).timestamp()),
    }
    return pyjwt.encode(payload, settings.jwt_secret, algorithm=settings.jwt_algorithm)


def decode_token(token: str, settings: Settings | None = None) -> Identity:
    settings = settings or get_settings()
    try:
        payload = pyjwt.decode(token, settings.jwt_secret, algorithms=[settings.jwt_algorithm])
    except pyjwt.ExpiredSignatureError as exc:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Your session expired. Please sign in again.") from exc
    except pyjwt.InvalidTokenError as exc:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Invalid session token.") from exc
    try:
        return Identity(
            employee_id=payload["sub"],
            company_id=payload["company_id"],
            role=payload.get("role", "employee"),
            name=payload.get("name", ""),
        )
    except KeyError as exc:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Token is missing identity claims.") from exc


_bearer = HTTPBearer(auto_error=False)


async def get_identity(
    credentials: HTTPAuthorizationCredentials | None = Depends(_bearer),
    settings: Settings = Depends(get_settings),
) -> Identity:
    if credentials is None or credentials.scheme.lower() != "bearer":
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Send an Authorization: Bearer <token> header.")
    return decode_token(credentials.credentials, settings)


async def require_manager(identity: Identity = Depends(get_identity)) -> Identity:
    if identity.role != "manager":
        raise HTTPException(status.HTTP_403_FORBIDDEN, "Only transport managers can use this endpoint.")
    return identity
