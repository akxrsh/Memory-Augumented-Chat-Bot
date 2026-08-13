import uuid
from datetime import datetime
from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordRequestForm
from backend.schemas.api_schemas import UserRegister, UserOut, Token, UserLogin
from backend.models.database import db_get_user_by_username, db_create_user
from backend.services.auth import get_password_hash, verify_password, create_access_token
from backend.utils.logger import logger

router = APIRouter(prefix="/auth", tags=["Authentication"])

@router.post("/register", response_model=UserOut, status_code=status.HTTP_201_CREATED)
async def register(user_data: UserRegister):
    """Registers a new user into the system."""
    existing_user = await db_get_user_by_username(user_data.username)
    if existing_user:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Username already registered"
        )
    
    user_id = str(uuid.uuid4())
    hashed_pwd = get_password_hash(user_data.password)
    
    try:
        user_doc = await db_create_user(
            user_id=user_id,
            username=user_data.username,
            password_hash=hashed_pwd,
            role=user_data.role
        )
        return user_doc
    except Exception as e:
        logger.error(f"Error registering user: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to register user"
        )

@router.post("/login", response_model=Token)
async def login(form_data: OAuth2PasswordRequestForm = Depends()):
    """OAuth2 password login flow returning a JWT token."""
    user = await db_get_user_by_username(form_data.username)
    if not user or not verify_password(form_data.password, user["password_hash"]):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    # Generate token
    token_data = {"sub": user["id"], "username": user["username"], "role": user["role"]}
    access_token = create_access_token(data=token_data)
    
    return {
        "access_token": access_token,
        "token_type": "bearer",
        "username": user["username"],
        "role": user["role"]
    }
