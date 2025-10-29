import uvicorn
from fastapi import FastAPI
from fastapi_sqlalchemy import DBSessionMiddleware
from starlette.middleware.cors import CORSMiddleware


from app.api.api_router import router
from app.core.config import settings
from app.helpers.exception_handler import CustomException, http_exception_handler
from app.middleware.auth_middleware import ValidateTokenMiddleware
from app.scheduler import scheduler_cron, schedule_delete_old_documents, schedule_file_uploads_processing

app = FastAPI(
    title=settings.PROJECT_NAME,
    docs_url="/docs",
    redoc_url='/re-docs',
    openapi_url=f"{settings.API_PREFIX}/openapi.json",
    description='''
        Base frame with FastAPI micro framework + Postgresql
            - Login/Register with JWT
            - Permission
            - CRUD User
            - Unit testing with Pytest
            - Dockerize
    '''
)

# Add CORS middleware first
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://genifast.pro", "http://localhost:5173", "http://103.157.218.224:5173","https://genadata.com","https://python.genadata.com"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.add_middleware(DBSessionMiddleware, db_url=settings.DATABASE_URL)
# app.add_middleware(
#     ValidateTokenMiddleware,
#     public_key=settings.ACCESS_TOKEN_PUBLIC_KEY,
#     expected_audience=["user_credentials", "socials"]
# )
app.include_router(router, prefix=settings.API_PREFIX)


@app.on_event("startup")
async def start_scheduler():
    await schedule_delete_old_documents()
    await schedule_file_uploads_processing()
    scheduler_cron.start()

app.add_exception_handler(CustomException, http_exception_handler)

if __name__ == '__main__':
    uvicorn.run(app, host="0.0.0.0", port=8000)