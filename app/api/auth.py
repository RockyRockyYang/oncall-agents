from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from app.api.deps import get_current_user
from app.core.security import create_access_token, hash_password, verify_password
from app.db.crud import create_user, get_user_by_username
from app.db.models import UserRow

router = APIRouter()


class RegisterRequest(BaseModel):
    username: str
    password: str


class LoginRequest(BaseModel):
    username: str
    password: str


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"


class UserResponse(BaseModel):
    id: str
    username: str


@router.post("/auth/register", response_model=UserResponse)
async def register(request: RegisterRequest) -> UserResponse:
    if await get_user_by_username(request.username):
        raise HTTPException(status_code=400, detail="用户名已被占用")
    user = await create_user(request.username, hash_password(request.password))
    return UserResponse(id=user.id, username=user.username)


@router.post("/auth/login", response_model=TokenResponse)
async def login(request: LoginRequest) -> TokenResponse:
    user = await get_user_by_username(request.username)
    # 用户不存在和密码错误统一报同一个错误，不让别人能靠报错信息探测出哪些用户名已注册
    if user is None or not verify_password(request.password, user.password_hash):
        raise HTTPException(status_code=401, detail="用户名或密码错误")
    return TokenResponse(access_token=create_access_token(user.id))


@router.get("/auth/me", response_model=UserResponse)
async def me(current_user: UserRow = Depends(get_current_user)) -> UserResponse:
    return UserResponse(id=current_user.id, username=current_user.username)
