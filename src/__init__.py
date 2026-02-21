from fastapi import FastAPI
from src.backend.router import api_router
app=FastAPI()
version='v1'
app.include_router(api_router,prefix=f'/api/{version}',tags=['classification'])