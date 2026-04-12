# main.py
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from config import PORT
from api.userApi import userRouter
from api.chatApi import chatRouter
from api.historyApi import historyRouter      # new
from api.quizApi import quizRouter            # new
from api.personalizationApi import personalizationRouter
from api.studyPlannerApi import studyPlannerRouter
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from database.mongodb import connect_db, close_db
from api.conceptOfDayApi import conceptRouter
from api.tutorialsApi import tutorialsRouter
from api.coachingApi import coachingRouter
from api.leaderboardApi import leaderboardRouter, update_leaderboard_points


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
app.include_router(personalizationRouter, prefix="/api")
app.include_router(studyPlannerRouter, prefix="/api")
app.include_router(conceptRouter, prefix="/api")
app.include_router(tutorialsRouter, prefix="/api")
app.include_router(coachingRouter, prefix="/api")
app.include_router(leaderboardRouter, prefix="/api")



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