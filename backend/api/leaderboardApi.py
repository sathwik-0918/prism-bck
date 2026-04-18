# api/leaderboardApi.py
# Calculates and stores user scores from ALL features
# Score = quiz performance + chat usage + concepts + study planner + tutorials + behavior
# Refreshes weekly/monthly scores on schedule

from database.mongodb import get_db, get_cloud_db

# In ALL leaderboard functions, use get_cloud_db() for:
# - db.leaderboard collection
# - db.studychat_users collection (for profile display)
# Keep get_db() for local data like personalization

from fastapi import APIRouter
# from database.mongodb import get_db
from datetime import datetime, timedelta
from typing import List

leaderboardRouter = APIRouter()


def get_week_start():
    today = datetime.utcnow()
    return (today - timedelta(days=today.weekday())).replace(
        hour=0, minute=0, second=0, microsecond=0
    ).isoformat()


def get_month_start():
    return datetime.utcnow().replace(
        day=1, hour=0, minute=0, second=0, microsecond=0
    ).isoformat()


def get_year_start():
    return datetime.utcnow().replace(
        month=1, day=1, hour=0, minute=0, second=0, microsecond=0
    ).isoformat()


async def calculate_user_score(userId: str, db, since: str = None) -> dict:
    """
    Calculates comprehensive score for a user.
    Components:
    - Quiz: 10 pts per quiz + score bonus
    - Chat: 2 pts per quality query
    - Concepts viewed: 5 pts per concept
    - Study planner: 20 pts for having a plan
    - Tutorials watched: 3 pts each
    - Streak bonus: extra pts for daily activity
    """
    score = 0
    breakdown = {}

    # ── QUIZ SCORE ─────────────────────────
    quiz_filter = {"userId": userId}
    if since:
        quiz_filter["completedAt"] = {"$gte": since}

    quizzes = await db.quizhistory.find(quiz_filter, {"scorePercent": 1, "correct": 1}).to_list(100)
    quiz_score = 0
    for q in quizzes:
        quiz_score += 10  # participation
        quiz_score += int(q.get("scorePercent", 0) * 0.5)  # score bonus (max 50 pts per quiz)
        if q.get("scorePercent", 0) == 100:
            quiz_score += 20  # perfect score bonus
    breakdown["quizScore"] = quiz_score
    score += quiz_score

    # ── CHAT SCORE ─────────────────────────
    session_filter = {"userId": userId}
    if since:
        session_filter["updatedAt"] = {"$gte": since}

    sessions = await db.sessions.find(session_filter, {"messages": 1}).to_list(100)
    chat_score = 0
    for s in sessions:
        msgs = s.get("messages", [])
        user_msgs = [m for m in msgs if m.get("role") == "user"]
        # 2 pts per legitimate query (not too short)
        quality_queries = [m for m in user_msgs if len(m.get("content", "")) > 10]
        chat_score += len(quality_queries) * 2
    breakdown["chatScore"] = min(chat_score, 200)  # cap at 200
    score += breakdown["chatScore"]

    # ── CONCEPT OF DAY ─────────────────────
    concept_filter = {"userId": userId}
    if since:
        concept_filter["viewedAt"] = {"$gte": since}

    concepts = await db.conceptviews.count_documents(concept_filter)
    concept_score = concepts * 5
    breakdown["conceptScore"] = concept_score
    score += concept_score

    # ── CONCEPT QUESTION ATTEMPTS ──────────
    cq_filter = {"userId": userId}
    if since:
        cq_filter["attemptedAt"] = {"$gte": since}

    cq_attempts = await db.conceptquestions.find(cq_filter).to_list(100)
    cq_score = 0
    for a in cq_attempts:
        cq_score += 5  # attempted
        if a.get("correct"):
            cq_score += 10  # bonus for correct
    breakdown["conceptQuestionScore"] = cq_score
    score += cq_score

    # ── STUDY PLANNER ──────────────────────
    plan = await db.studyplans.find_one({"userId": userId})
    if plan:
        planner_score = 20  # has a plan
        completed_tasks = sum(1 for v in plan.get("taskProgress", {}).values() if v)
        planner_score += completed_tasks * 3
        breakdown["plannerScore"] = planner_score
        score += planner_score
    else:
        breakdown["plannerScore"] = 0

    # ── TUTORIALS ──────────────────────────
    tut_filter = {"userId": userId}
    if since:
        tut_filter["viewedAt"] = {"$gte": since}

    tutorials = await db.tutorialviews.count_documents(tut_filter)
    tut_score = tutorials * 3
    breakdown["tutorialScore"] = min(tut_score, 60)
    score += breakdown["tutorialScore"]

    # ── BEHAVIOR PENALTY ───────────────────
    # penalize vulgar/inappropriate queries
    all_sessions = await db.sessions.find({"userId": userId}, {"messages": 1}).to_list(50)
    blocked_count = 0
    bad_words = ["porn", "sex", "drug", "hack", "weapon", "nude"]
    for s in all_sessions:
        for m in s.get("messages", []):
            if m.get("role") == "user":
                content_lower = m.get("content", "").lower()
                if any(w in content_lower for w in bad_words):
                    blocked_count += 1
    penalty = blocked_count * 20
    breakdown["penalty"] = -penalty
    score = max(0, score - penalty)

    return {"total": score, "breakdown": breakdown}


async def update_leaderboard_points(userId: str, action: str, value: int, db):
    """Updates points in CLOUD leaderboard."""
    db_cloud = get_cloud_db()

    points_map = {
        "quiz": min(value * 0.5 + 10, 60),
        "chat": 2,
        "concept": 5,
        "tutorial": 3,
        "planner": 20,
        "battle": min(value * 0.6 + 15, 75),
    }

    if action == "conceptQuestion":
        points = value
    else:
        points = int(points_map.get(action, 0))

    # even if points is 0, we still want to upsert to ensure the user exists in the DB

    now = datetime.utcnow().isoformat()

    await db_cloud.leaderboard.update_one(
        {"userId": userId},
        {
            "$inc": {
                "allTimeScore": points,
                "weeklyScore": points,
                "monthlyScore": points,
                "yearlyScore": points,
            },
            "$set": {"lastActive": now},
            "$setOnInsert": {"joinedAt": now}
        },
        upsert=True
    )


@leaderboardRouter.get("/leaderboard/{period}")
async def getLeaderboard(period: str, examTarget: str = ""):
    """Gets leaderboard from CLOUD so all users see each other."""
    db_cloud = get_cloud_db()

    score_field = {
        "all-time": "allTimeScore",
        "weekly": "weeklyScore",
        "monthly": "monthlyScore",
        "yearly": "yearlyScore"
    }.get(period, "allTimeScore")

    cursor = db_cloud.leaderboard.find({}, {"_id": 0}).sort(score_field, -1).limit(100)
    entries = await cursor.to_list(length=100)
    user_ids = [entry.get("userId") for entry in entries if entry.get("userId")]
    cloud_profiles = await db_cloud.studychat_users.find(
        {"userId": {"$in": user_ids}},
        {"_id": 0, "userId": 1, "displayName": 1, "firstName": 1, "profileImageUrl": 1, "examTarget": 1}
    ).to_list(length=len(user_ids))
    profiles_by_id = {p.get("userId"): p for p in cloud_profiles}

    enriched = []
    for i, entry in enumerate(entries):
        profile = profiles_by_id.get(entry.get("userId"), {})
        display_name = (
            entry.get("firstName") or
            profile.get("displayName") or
            profile.get("firstName") or
            "User"
        )

        enriched.append({
            "rank": i + 1,
            "userId": entry["userId"],
            "firstName": display_name,
            "lastName": "",
            "profileImageUrl": entry.get("profileImageUrl") or profile.get("profileImageUrl", ""),
            "examTarget": entry.get("examTarget") or profile.get("examTarget", ""),
            "score": entry.get(score_field, 0),
            "allTimeScore": entry.get("allTimeScore", 0),
            "lastActive": entry.get("lastActive", ""),
            "breakdown": entry.get("breakdown", {})
        })

    return {"message": "leaderboard", "payload": enriched, "period": period}


@leaderboardRouter.post("/leaderboard/add-user/{userId}")
async def addToLeaderboard(userId: str, firstName: str = "", profileImageUrl: str = "", examTarget: str = ""):
    """
    Adds user to cloud leaderboard on first login.
    This makes all users visible to each other for friend discovery.
    """
    db_cloud = get_cloud_db()
    now_iso = datetime.utcnow().isoformat()

    # store in cloud leaderboard
    await db_cloud.leaderboard.update_one(
        {"userId": userId},
        {
            "$set": {
                "userId": userId,
                "firstName": firstName,
                "profileImageUrl": profileImageUrl,
                "examTarget": examTarget,
                "lastActive": now_iso
            },
            "$setOnInsert": {
                "allTimeScore": 0,
                "weeklyScore": 0,
                "monthlyScore": 0,
                "yearlyScore": 0,
                "joinedAt": now_iso
            }
        },
        upsert=True
    )

    # also store in cloud chat users for search
    await db_cloud.studychat_users.update_one(
        {"userId": userId},
        {"$set": {
            "userId": userId,
            "displayName": firstName,
            "firstName": firstName,
            "profileImageUrl": profileImageUrl,
            "examTarget": examTarget,
            "updatedAt": now_iso
        }},
        upsert=True
    )

    return {"message": "ok"}


@leaderboardRouter.get("/leaderboard/user/{userId}")
async def getUserRank(userId: str):
    """Gets a specific user's rank and score across all periods."""
    db_cloud = get_cloud_db()
    entry = await db_cloud.leaderboard.find_one({"userId": userId}, {"_id": 0})
    if not entry:
        return {"message": "not found", "payload": None}

    # calculate ranks
    all_time_rank = await db_cloud.leaderboard.count_documents(
        {"allTimeScore": {"$gt": entry.get("allTimeScore", 0)}}
    ) + 1
    weekly_rank = await db_cloud.leaderboard.count_documents(
        {"weeklyScore": {"$gt": entry.get("weeklyScore", 0)}}
    ) + 1

    return {
        "message": "rank",
        "payload": {
            **entry,
            "allTimeRank": all_time_rank,
            "weeklyRank": weekly_rank
        }
    }
