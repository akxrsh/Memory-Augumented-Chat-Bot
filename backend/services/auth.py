from datetime import datetime, timedelta
from typing import Optional, Union, Dict, Any
import bcrypt
from jose import jwt, JWTError
from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from backend.config.settings import settings
from backend.models.database import db_get_user_by_id
from backend.utils.logger import logger

# OAuth2 Scheme
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="api/v1/auth/login")

def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verifies that the plain password matches the hashed password."""
    try:
        return bcrypt.checkpw(plain_password.encode('utf-8'), hashed_password.encode('utf-8'))
    except Exception as e:
        logger.error(f"Password verification error: {e}")
        return False

def get_password_hash(password: str) -> str:
    """Generates a bcrypt hash of the password."""
    # Generate salt and hash
    salt = bcrypt.gensalt()
    hashed = bcrypt.hashpw(password.encode('utf-8'), salt)
    return hashed.decode('utf-8')

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
    """Generates a JWT access token containing the payload data."""
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)
    
    to_encode.update({"exp": expire})
    try:
        encoded_jwt = jwt.encode(to_encode, settings.JWT_SECRET, algorithm=settings.JWT_ALGORITHM)
        return encoded_jwt
    except Exception as e:
        logger.error(f"Error generating JWT: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to generate authentication token"
        )

async def get_current_user(token: str = Depends(oauth2_scheme)) -> Dict[str, Any]:
    """FastAPI dependency to retrieve the currently logged in user based on the JWT token."""
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, settings.JWT_SECRET, algorithms=[settings.JWT_ALGORITHM])
        user_id: str = payload.get("sub")
        if user_id is None:
            raise credentials_exception
    except JWTError:
        raise credentials_exception
        
    user = await db_get_user_by_id(user_id)
    if user is None:
        raise credentials_exception
        
    return user

def require_role(roles: list[str]):
    """Returns a dependency function checking if user has appropriate access role."""
    def dependency(current_user: Dict[str, Any] = Depends(get_current_user)):
        user_role = current_user.get("role", "user")
        if user_role not in roles:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="You do not have permission to access this resource"
            )
        return current_user
    return dependency
