# main.py
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from config import PORT
from api.userApi import userRouter
from api.chatApi import chatRouter
from api.historyApi import historyRouter      # new
from api.quizApi import quizRouter            # new
from database.mongodb import connect_db, close_db

@asynccontextmanager
async def lifespan(app: FastAPI):
    # startup — like app.listen() callback in Express
    await connect_db()
    yield
    # shutdown
    await close_db()

app = FastAPI(title="Prism API", version="1.0.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(userRouter, prefix="/api")
app.include_router(chatRouter, prefix="/api")
app.include_router(historyRouter, prefix="/api")
app.include_router(quizRouter, prefix="/api")

@app.get("/health")
def health_check():
    return {"status": "ok", "service": "Prism Backend"}