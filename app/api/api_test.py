from fastapi import APIRouter

from app.services.category_service import get_all_categories

router = APIRouter()


@router.get("")
async def get():
    return 'test'


@router.get("/categories")
def read_categories():
    return get_all_categories()
