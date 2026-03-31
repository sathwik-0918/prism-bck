from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from config import PORT
from api.userApi import userRouter

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(userRouter, prefix="/api")

@app.get("/health")
def health_check():
    return { "status": "ok" }