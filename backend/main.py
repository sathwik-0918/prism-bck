# main.py
# FastAPI application entry point
# registers all routers and middleware

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from config import PORT
from api.userApi import userRouter
from api.chatApi import chatRouter         # ← add this

app = FastAPI(title="Prism API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# register routers
app.include_router(userRouter, prefix="/api")
app.include_router(chatRouter, prefix="/api")   # ← add this

@app.get("/health")
def health_check():
    return {"status": "ok", "service": "Prism Backend"}