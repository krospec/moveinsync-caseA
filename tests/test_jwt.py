"""JWT behaviour: no token, garbage, expired - and identity strictly from claims."""
from app.config import get_settings
from tests.conftest import auth_header, auth_token

import jwt as pyjwt


async def test_missing_token_is_401(client):
    resp = await client.post("/chat", json={"message": "hi"})
    assert resp.status_code == 401


async def test_garbage_token_is_401(client):
    resp = await client.post(
        "/chat", json={"message": "hi"}, headers={"Authorization": "Bearer not-a-jwt"}
    )
    assert resp.status_code == 401


async def test_expired_token_is_401(client):
    settings = get_settings()
    now = 1_000_000  # 1970-ish: long expired
    token = pyjwt.encode(
        {
            "sub": "acme-rhea", "company_id": "acme", "role": "employee", "name": "Rhea",
            "iat": now, "exp": now + 60,
        },
        settings.jwt_secret,
        algorithm=settings.jwt_algorithm,
    )
    resp = await client.post(
        "/chat", json={"message": "hi"}, headers=auth_header(token)
    )
    assert resp.status_code == 401


async def test_wrong_secret_is_401(client):
    settings = get_settings()
    token = pyjwt.encode(
        {"sub": "acme-rhea", "company_id": "acme", "role": "employee", "name": "Rhea",
         "iat": 1760000000, "exp": 9999999999},
        "some-other-secret",
        algorithm=settings.jwt_algorithm,
    )
    resp = await client.post("/chat", json={"message": "hi"}, headers=auth_header(token))
    assert resp.status_code == 401


async def test_token_carries_company_and_role(client):
    token = await auth_token(client, "globex-mgr")
    resp = await client.get("/manager/requests", headers=auth_header(token))
    assert resp.status_code == 200

    employee_token = await auth_token(client, "globex-arjun")
    resp = await client.get("/manager/requests", headers=auth_header(employee_token))
    assert resp.status_code == 403
