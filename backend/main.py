# main.py
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import socketio

from config import PORT
from api.userApi import userRouter
from api.chatApi import chatRouter
from api.historyApi import historyRouter      # new
from api.quizApi import quizRouter            # new
from api.personalizationApi import personalizationRouter
from api.studyPlannerApi import studyPlannerRouter
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from api.conceptOfDayApi import conceptRouter
from api.tutorialsApi import tutorialsRouter
from api.coachingApi import coachingRouter
from api.leaderboardApi import leaderboardRouter, update_leaderboard_points
from api.studyChatRestApi import studyChatRouter
from api.studyChatApi import sio

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

# Register all HTTP routers
for router in [userRouter, chatRouter, historyRouter, quizRouter,
               personalizationRouter, studyPlannerRouter, conceptRouter,
               leaderboardRouter, coachingRouter, tutorialsRouter, studyChatRouter]:
    app.include_router(router, prefix="/api")



@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    try:
        body = await request.json()
    except Exception:
        body = None
    print(f"Validation error: {exc.errors()}\nBody: {body}")
    return JSONResponse(status_code=422, content={"detail": exc.errors()})

@app.get("/health")
def health_check():
    return {"status": "ok", "service": "Prism Backend"}

# ── WRAP WITH SOCKET.IO ──────────────────────────────────────────────────
# This is the key — wrap FastAPI app with Socket.IO ASGI middleware
# Socket.IO handles /socket.io/* routes, FastAPI handles everything else
app = socketio.ASGIApp(
    socketio_server=sio,
    other_asgi_app=app,
    socketio_path="socket.io"
)