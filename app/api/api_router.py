from fastapi import APIRouter

from app.api import api_test, api_document

router = APIRouter()

router.include_router(api_test.router, tags=["test"], prefix="/test")
router.include_router(api_document.router, tags=["file"], prefix="/file")
