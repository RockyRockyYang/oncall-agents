from fastapi import Depends, HTTPException
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from app.core.security import decode_access_token
from app.db.crud import get_user_by_id
from app.db.models import UserRow

security = HTTPBearer()


async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)) -> UserRow:
    user_id = decode_access_token(credentials.credentials)
    if user_id is None:
        raise HTTPException(status_code=401, detail="无效或已过期的登录凭证")
    user = await get_user_by_id(user_id)
    if user is None:
        raise HTTPException(status_code=401, detail="无效或已过期的登录凭证")
    return user
