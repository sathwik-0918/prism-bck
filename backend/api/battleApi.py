# api/battleApi.py
# Real-time Battle Rooms for competitive quiz
# Uses existing Socket.IO server from studyChatApi.py
# Features: live leaderboard, timer sync, AI quiz, anti-cheat

from fastapi import APIRouter
from pydantic import BaseModel
from typing import List, Optional, Dict
from database.mongodb import get_db
from rag.nodes import main_llm, vector_store
from langchain_core.messages import SystemMessage, HumanMessage
from datetime import datetime, timedelta
import uuid, json, re, asyncio
from api.quizApi import quiz_llm, parse_quiz_response, parse_quiz_flexible

battleRouter = APIRouter()

def now():
    return datetime.utcnow().isoformat() + "Z"


# ── BATTLE SOCKET EVENTS (added to studyChatApi.py sio) ──────────────────
# These events get registered on the same sio instance

def register_battle_events(sio):
    """
    Register all battle room Socket.IO events.
    Called from main.py after sio is created.
    """

    @sio.event
    async def create_battle_room(sid, data):
        """
        Create a new battle room.
        data: {userId, roomName, topic, difficulty, questionCount,
               isPrivate, examTarget, inviteCode}
        """
        from api.studyChatApi import socket_to_user
        user_id = socket_to_user.get(sid)
        if not user_id:
            return

        from database.mongodb import get_db
        db = get_db()

        room_id = str(uuid.uuid4())[:8].upper()  # short code like "ABC12345"
        invite_code = data.get("inviteCode") or str(uuid.uuid4())[:6].upper()
        use_timer = bool(data.get("useTimer", True))
        requested_time = int(data.get("timePerQuestion", 30) or 30)
        time_per_question = max(5, min(requested_time, 120)) if use_timer else 30

        room = {
            "roomId": room_id,
            "inviteCode": invite_code,
            "roomName": data.get("roomName", f"{user_id}'s Room"),
            "hostId": user_id,
            "topic": data.get("topic", "Mixed"),
            "difficulty": data.get("difficulty", "medium"),
            "questionCount": min(int(data.get("questionCount", 10)), 20),
            "examTarget": data.get("examTarget", "JEE"),
            "isPrivate": data.get("isPrivate", False),
            "status": "waiting",    # waiting | generating | countdown | active | finished
            "members": [{
                "userId": user_id,
                "joinedAt": now(),
                "score": 0,
                "answers": {},
                "rank": 1
            }],
            "questions": [],         # generated when game starts
            "currentQuestion": 0,
            "startTime": None,
            "questionStartTime": None,
            "useTimer": use_timer,
            "timePerQuestion": time_per_question,   # seconds
            "createdAt": now()
        }

        await db.battle_rooms.insert_one(room)
        room.pop("_id", None)

        await sio.enter_room(sid, f"battle_{room_id}")
        await sio.emit("room_created", {
            "roomId": room_id,
            "inviteCode": invite_code,
            "room": room
        }, to=sid)

        # announce to public lobby if not private
        if not data.get("isPrivate"):
            await sio.emit("public_room_added", {
                "roomId": room_id,
                "roomName": room["roomName"],
                "topic": room["topic"],
                "difficulty": room["difficulty"],
                "memberCount": 1
            }, room="battle_lobby")

        print(f"[Battle] Room created: {room_id} by {user_id}")


    @sio.event
    async def join_battle_room(sid, data):
        """
        Join a battle room by roomId or inviteCode.
        data: {userId, roomId} OR {userId, inviteCode}
        """
        from api.studyChatApi import socket_to_user
        user_id = socket_to_user.get(sid)
        if not user_id:
            return

        from database.mongodb import get_db
        db = get_db()

        # find room
        query = {}
        if data.get("roomId"):
            query["roomId"] = data["roomId"]
        elif data.get("inviteCode"):
            query["inviteCode"] = data["inviteCode"].upper()

        room = await db.battle_rooms.find_one(query)

        if not room:
            await sio.emit("battle_error", {"message": "Room not found"}, to=sid)
            return

        if room["status"] != "waiting":
            await sio.emit("battle_error", {"message": "Game already started"}, to=sid)
            return

        room_id = room["roomId"]

        # check if already in room
        if not any(m["userId"] == user_id for m in room.get("members", [])):
            await db.battle_rooms.update_one(
                {"roomId": room_id},
                {"$push": {"members": {
                    "userId": user_id,
                    "joinedAt": now(),
                    "score": 0,
                    "answers": {},
                    "rank": len(room["members"]) + 1
                }}}
            )

        await sio.enter_room(sid, f"battle_{room_id}")

        # notify all in room
        updated_room = await db.battle_rooms.find_one({"roomId": room_id}, {"_id": 0})
        await sio.emit("room_updated", {"room": updated_room}, room=f"battle_{room_id}")
        await sio.emit("player_joined", {
            "userId": user_id,
            "roomId": room_id
        }, room=f"battle_{room_id}")

        print(f"[Battle] {user_id} joined room {room_id}")


    @sio.event
    async def start_battle(sid, data):
        """
        Host starts the battle. Generates questions, begins countdown.
        data: {roomId}
        """
        from api.studyChatApi import socket_to_user
        user_id = socket_to_user.get(sid)
        room_id = data.get("roomId")

        if not user_id or not room_id:
            return

        from database.mongodb import get_db
        db = get_db()

        room = await db.battle_rooms.find_one({"roomId": room_id})
        if not room or room["hostId"] != user_id:
            await sio.emit("battle_error", {"message": "Only host can start"}, to=sid)
            return

        if room.get("status") != "waiting":
            await sio.emit("battle_error", {"message": "Battle already starting or active"}, to=sid)
            return

        if len(room.get("members", [])) < 1:
            await sio.emit("battle_error", {"message": "Need at least 1 player"}, to=sid)
            return

        # Acquire a start lock so repeated clicks/events do not trigger
        # parallel question generation for the same room.
        locked = await db.battle_rooms.update_one(
            {"roomId": room_id, "hostId": user_id, "status": "waiting"},
            {"$set": {"status": "generating"}}
        )
        if locked.modified_count == 0:
            await sio.emit("battle_error", {"message": "Battle already starting"}, to=sid)
            return

        # generate questions using existing quiz API logic
        await sio.emit("generating_questions", {
            "roomId": room_id,
            "message": "AI is generating questions from your study materials..."
        }, room=f"battle_{room_id}")

        questions = await generate_battle_questions(
            room["topic"], room["examTarget"],
            room["difficulty"], room["questionCount"]
        )

        if not questions:
            await db.battle_rooms.update_one(
                {"roomId": room_id},
                {"$set": {"status": "waiting"}}
            )
            await sio.emit("battle_error", {
                "message": "Failed to generate questions. Try again."
            }, room=f"battle_{room_id}")
            return

        # 5 second countdown
        await db.battle_rooms.update_one(
            {"roomId": room_id},
            {"$set": {"status": "countdown", "questions": questions}}
        )

        await sio.emit("battle_countdown", {"roomId": room_id, "count": 5},
                       room=f"battle_{room_id}")

        for i in range(4, 0, -1):
            await asyncio.sleep(1)
            await sio.emit("battle_countdown", {"roomId": room_id, "count": i},
                           room=f"battle_{room_id}")

        # start battle
        # Use explicit UTC timestamp format (with Z) so frontend timer parsing
        # does not treat it as local time and instantly expire Q1.
        start_time = datetime.utcnow().isoformat() + "Z"
        await db.battle_rooms.update_one(
            {"roomId": room_id},
            {"$set": {
                "status": "active",
                "currentQuestion": 0,
                "startTime": start_time,
                "questionStartTime": start_time
            }}
        )

        await sio.emit("battle_started", {
            "roomId": room_id,
            "question": questions[0],
            "questionIndex": 0,
            "totalQuestions": len(questions),
            "useTimer": room.get("useTimer", True),
            "timePerQuestion": room["timePerQuestion"],
            "startTime": start_time
        }, room=f"battle_{room_id}")

        print(f"[Battle] Battle started: {room_id} with {len(questions)} questions")

        # schedule question progression
        asyncio.create_task(
            progress_questions(
                sio,
                db,
                room_id,
                len(questions),
                room["timePerQuestion"],
                room.get("useTimer", True)
            )
        )


    @sio.event
    async def submit_battle_answer(sid, data):
        """
        Submit answer for current question.
        data: {roomId, questionIndex, selectedAnswer, timeTaken}
        Anti-cheat: validate timing, question index
        """
        from api.studyChatApi import socket_to_user
        user_id = socket_to_user.get(sid)
        room_id = data.get("roomId")
        question_idx = data.get("questionIndex", 0)
        selected_answer = data.get("selectedAnswer")
        room_time_limit = 30

        if not user_id or not room_id:
            return

        from database.mongodb import get_db
        db = get_db()

        room = await db.battle_rooms.find_one({"roomId": room_id})
        if not room or room["status"] != "active":
            return
        room_time_limit = max(int(room.get("timePerQuestion", 30) or 30), 1)
        time_taken = min(float(data.get("timeTaken", room_time_limit)), room_time_limit)

        # anti-cheat: verify correct question index
        if question_idx != room.get("currentQuestion", 0):
            return

        questions = room.get("questions", [])
        if question_idx >= len(questions):
            return

        # Check if already answered
        for member in room.get("members", []):
            if member["userId"] == user_id:
                if str(question_idx) in member.get("answers", {}):
                    return
                break

        is_correct = (selected_answer == questions[question_idx].get("answer"))
        correct_answer = questions[question_idx].get("answer")

        # calculate points (faster = more points, max 100 per question)
        if is_correct:
            speed_bonus = max(0, room_time_limit - time_taken)
            points = int(50 + (speed_bonus / room_time_limit) * 50)  # 50-100 points
        else:
            points = 0

        # update member score
        await db.battle_rooms.update_one(
            {"roomId": room_id, "members.userId": user_id},
            {
                "$inc": {"members.$.score": points},
                "$set": {
                    f"members.$.answers.{question_idx}": {
                        "selected": selected_answer,
                        "correct": is_correct,
                        "points": points,
                        "timeTaken": time_taken
                    }
                }
            }
        )

        # get updated room and broadcast leaderboard
        updated = await db.battle_rooms.find_one({"roomId": room_id})
        leaderboard = build_leaderboard(updated.get("members", []))

        await sio.emit("leaderboard_update", {
            "roomId": room_id,
            "leaderboard": leaderboard,
            "lastAnswer": {
                "userId": user_id,
                "isCorrect": is_correct,
                "points": points
            }
        }, room=f"battle_{room_id}")

        # personal result
        await sio.emit("answer_result", {
            "isCorrect": is_correct,
            "correctAnswer": correct_answer,
            "explanation": questions[question_idx].get("explanation", ""),
            "pointsEarned": points
        }, to=sid)

    @sio.event
    async def request_next_question(sid, data):
        """Allow host to move to next question without waiting full timer."""
        from api.studyChatApi import socket_to_user
        user_id = socket_to_user.get(sid)
        room_id = data.get("roomId")
        if not user_id or not room_id:
            return

        from database.mongodb import get_db
        db = get_db()
        room = await db.battle_rooms.find_one({"roomId": room_id})
        if not room or room.get("hostId") != user_id or room.get("status") != "active":
            return

        await advance_to_next_question(sio, db, room_id)


    @sio.event
    async def join_battle_lobby(sid, data):
        """Join public lobby to see available rooms."""
        await sio.enter_room(sid, "battle_lobby")

        from database.mongodb import get_db
        db = get_db()

        rooms = await db.battle_rooms.find(
            {"isPrivate": False, "status": "waiting"},
            {"_id": 0, "questions": 0}
        ).sort("createdAt", -1).limit(20).to_list(20)

        await sio.emit("lobby_rooms", {"rooms": rooms}, to=sid)


    @sio.event
    async def leave_battle_room(sid, data):
        """Leave a battle room."""
        from api.studyChatApi import socket_to_user
        user_id = socket_to_user.get(sid)
        room_id = data.get("roomId")
        if user_id and room_id:
            await sio.leave_room(sid, f"battle_{room_id}")
            await sio.emit("player_left", {
                "userId": user_id, "roomId": room_id
            }, room=f"battle_{room_id}")


    @sio.event
    async def delete_battle_room(sid, data):
        """Host can delete room before battle starts."""
        from api.studyChatApi import socket_to_user
        user_id = socket_to_user.get(sid)
        room_id = data.get("roomId")
        if not user_id or not room_id:
            return

        from database.mongodb import get_db
        db = get_db()
        room = await db.battle_rooms.find_one({"roomId": room_id})
        if not room:
            await sio.emit("battle_error", {"message": "Room not found"}, to=sid)
            return
        if room.get("hostId") != user_id:
            await sio.emit("battle_error", {"message": "Only host can delete room"}, to=sid)
            return
        if room.get("status") != "waiting":
            await sio.emit("battle_error", {"message": "Cannot delete after battle starts"}, to=sid)
            return

        await db.battle_rooms.delete_one({"roomId": room_id})
        await sio.emit("room_deleted", {"roomId": room_id}, room=f"battle_{room_id}")
        await sio.emit("public_room_removed", {"roomId": room_id}, room="battle_lobby")


async def progress_questions(sio, db, room_id: str, total_questions: int, time_per_q: int, use_timer: bool):
    """
    Automatically advances questions after timer expires.
    Runs as background task.
    """
    if not use_timer:
        return

    while True:
        await asyncio.sleep(0.5)
        room = await db.battle_rooms.find_one({"roomId": room_id})
        if not room or room.get("status") != "active":
            break

        current_idx = int(room.get("currentQuestion", 0))
        if current_idx >= total_questions - 1:
            q_start_str = room.get("questionStartTime")
            if not q_start_str:
                continue
            elapsed = (datetime.utcnow() - datetime.fromisoformat(q_start_str.replace("Z", ""))).total_seconds()
            if elapsed >= time_per_q:
                await end_battle(sio, db, room_id)
                break
            continue

        q_start_str = room.get("questionStartTime")
        if not q_start_str:
            continue

        elapsed = (datetime.utcnow() - datetime.fromisoformat(q_start_str.replace("Z", ""))).total_seconds()
        if elapsed >= time_per_q:
            await advance_to_next_question(sio, db, room_id)


def _all_members_answered(members: list, question_idx: int) -> bool:
    key = str(question_idx)
    for member in members:
        answers = member.get("answers", {})
        if key not in answers and question_idx not in answers:
            return False
    return len(members) > 0


async def advance_to_next_question(sio, db, room_id: str):
    """Move room to next question once; ends battle at last question."""
    room = await db.battle_rooms.find_one({"roomId": room_id})
    if not room or room.get("status") != "active":
        return

    questions = room.get("questions", [])
    current_idx = int(room.get("currentQuestion", 0))
    next_idx = current_idx + 1
    if next_idx >= len(questions):
        await end_battle(sio, db, room_id)
        return

    q_start = datetime.utcnow().isoformat() + "Z"
    await db.battle_rooms.update_one(
        {"roomId": room_id, "status": "active", "currentQuestion": current_idx},
        {"$set": {"currentQuestion": next_idx, "questionStartTime": q_start}}
    )

    updated = await db.battle_rooms.find_one({"roomId": room_id}, {"_id": 0})
    if not updated or int(updated.get("currentQuestion", 0)) != next_idx:
        return

    await sio.emit("next_question", {
        "roomId": room_id,
        "question": questions[next_idx],
        "questionIndex": next_idx,
        "totalQuestions": len(questions),
        "useTimer": updated.get("useTimer", True),
        "timePerQuestion": updated.get("timePerQuestion", 30),
        "questionStartTime": q_start
    }, room=f"battle_{room_id}")


async def end_battle(sio, db, room_id: str):
    """End the battle and calculate final results."""
    room = await db.battle_rooms.find_one({"roomId": room_id})
    if not room:
        return

    members = room.get("members", [])
    leaderboard = build_leaderboard(members)

    # save to battle history (include per-user answer maps for review screen)
    member_answers = {
        m["userId"]: m.get("answers", {})
        for m in members
    }
    battle_result = {
        "resultId": str(uuid.uuid4()),
        "roomId": room_id,
        "roomName": room.get("roomName"),
        "topic": room.get("topic"),
        "difficulty": room.get("difficulty"),
        "examTarget": room.get("examTarget"),
        "leaderboard": leaderboard,
        "questions": room.get("questions", []),
        "memberAnswers": member_answers,
        "startTime": room.get("startTime"),
        "endTime": now(),
        "memberIds": [m["userId"] for m in members]
    }
    await db.battle_history.insert_one(battle_result)

    # update room status
    await db.battle_rooms.update_one(
        {"roomId": room_id},
        {"$set": {"status": "finished"}}
    )

    # update personalization for each member
    for member in members:
        user_answers = member.get("answers", {})
        questions = room.get("questions", [])
        correct = sum(1 for ans in user_answers.values() if ans.get("correct"))
        total = len(questions)

        await db.personalization.update_one(
            {"userId": member["userId"]},
            {
                "$inc": {
                    "battleRoomsPlayed": 1,
                    "battleCorrectAnswers": correct
                },
                "$set": {"lastBattleAt": now()}
            },
            upsert=True
        )

        # update leaderboard
        from api.leaderboardApi import update_leaderboard_points
        score_pct = int((correct / max(total, 1)) * 100)
        await update_leaderboard_points(
            member["userId"], "quiz", score_pct, db
        )

    await sio.emit("battle_ended", {
        "roomId": room_id,
        "leaderboard": leaderboard,
        "topic": room.get("topic"),
        "totalQuestions": len(room.get("questions", []))
    }, room=f"battle_{room_id}")

    print(f"[Battle] Battle ended: {room_id}")


def build_leaderboard(members: list) -> list:
    """Build sorted leaderboard from member data."""
    sorted_members = sorted(members, key=lambda m: m.get("score", 0), reverse=True)
    result = []
    for i, m in enumerate(sorted_members):
        answers = m.get("answers", {})
        correct = sum(1 for a in answers.values() if a.get("correct"))
        result.append({
            "rank": i + 1,
            "userId": m["userId"],
            "score": m.get("score", 0),
            "correctAnswers": correct,
            "totalAnswered": len(answers)
        })
    return result


async def generate_battle_questions(
    topic: str, exam_target: str, difficulty: str, count: int
) -> list:
    """
    Generate battle questions using same pipeline as quiz for speed/stability.
    """
    results = vector_store.query(
        query_text=f"{topic} {exam_target} questions",
        top_k=8
    )
    context = "\n\n".join([
        r["metadata"].get("text", "")
        for r in results if r["metadata"]
    ])[:3000]

    system_prompt = f"""You are an expert {exam_target} question setter.
Generate exactly {count} multiple choice questions on '{topic}'.
Difficulty level: {difficulty}

You MUST format EVERY question EXACTLY like this (no bold, no markdown, no numbering):

Q: [question text]
A) [option]
B) [option]
C) [option]
D) [option]
Answer: [A or B or C or D]
Explanation: [one line explanation]

Rules:
- Start each question with "Q: "
- Each option starts with "A) ", "B) ", "C) ", "D) "
- Answer line must be "Answer: " followed by ONLY the letter
- Explanation line must start with "Explanation: "
- Separate questions with a blank line
- Use ONLY the provided context"""

    try:
        response = await asyncio.to_thread(
            quiz_llm.invoke,
            [
                SystemMessage(content=system_prompt),
                HumanMessage(
                    content=f"Context:\n{context}\n\nGenerate {count} {difficulty} MCQ questions on '{topic}' now."
                )
            ]
        )

        text = response.content.strip()
        text = re.sub(r'```json\s*', '', text)
        text = re.sub(r'```\s*', '', text)
        parsed = parse_quiz_response(text)
        if not parsed:
            parsed = parse_quiz_flexible(text)
        if not parsed:
            parsed = parse_battle_questions_flexible(text)
        if parsed:
            return parsed[:count]
    except Exception as e:
        print(f"[Battle] Question generation error: {e}")

    return []


def parse_battle_questions_flexible(text: str) -> list:
    """
    Fallback parser for LLM outputs like:
    Q: ...
    A) ...
    B) ...
    C) ...
    D) ...
    Answer: A
    Explanation: ...
    """
    questions = []
    blocks = re.split(r'\n(?=Q\s*[:.])', text)
    for block in blocks:
        block = block.strip()
        if not block or not re.match(r'^Q\s*[:.]', block):
            continue
        lines = [ln.strip() for ln in block.splitlines() if ln.strip()]
        if not lines:
            continue

        q_text = re.sub(r'^Q\s*[:.]\s*', '', lines[0]).strip()
        options = {}
        answer = ""
        explanation = ""
        for ln in lines[1:]:
            opt = re.match(r'^([A-D])\)\s*(.+)$', ln)
            if opt:
                options[opt.group(1)] = opt.group(2).strip()
                continue
            ans = re.match(r'^Answer\s*:\s*([A-D])', ln, re.IGNORECASE)
            if ans:
                answer = ans.group(1).upper()
                continue
            exp = re.match(r'^Explanation\s*:\s*(.+)$', ln, re.IGNORECASE)
            if exp:
                explanation = exp.group(1).strip()

        if q_text and len(options) >= 2:
            questions.append({
                "question": q_text,
                "options": options,
                "answer": answer or "A",
                "explanation": explanation
            })
    return questions


# ── REST ENDPOINTS ────────────────────────────────────────────────────────

@battleRouter.get("/battle/rooms/public")
async def getPublicRooms():
    """Get all waiting public rooms."""
    db = get_db()
    rooms = await db.battle_rooms.find(
        {"isPrivate": False, "status": "waiting"},
        {"_id": 0, "questions": 0}
    ).sort("createdAt", -1).limit(20).to_list(20)
    return {"message": "rooms", "payload": rooms}


@battleRouter.get("/battle/room/{roomId}")
async def getRoom(roomId: str):
    """Get room details."""
    db = get_db()
    room = await db.battle_rooms.find_one(
        {"roomId": roomId}, {"_id": 0, "questions": 0}
    )
    return {"message": "room", "payload": room}


@battleRouter.get("/battle/history/{userId}")
async def getBattleHistory(userId: str):
    """Get battle history for sidebar."""
    db = get_db()
    history = await db.battle_history.find(
        {"memberIds": userId},
        {"_id": 0, "questions": 0}
    ).sort("endTime", -1).limit(30).to_list(30)

    # convert _id
    for h in history:
        if "_id" in h:
            h["_id"] = str(h["_id"])

    return {"message": "history", "payload": history}


@battleRouter.get("/battle/history/{userId}/{resultId}")
async def getBattleHistoryDetail(userId: str, resultId: str):
    """Get one battle with full question + per-user answers for review."""
    db = get_db()
    result = await db.battle_history.find_one(
        {"resultId": resultId, "memberIds": userId},
        {"_id": 0}
    )
    if not result:
        return {"message": "not found", "payload": None}
    return {"message": "battle history detail", "payload": result}


@battleRouter.get("/battle/leaderboard")
async def getBattleLeaderboard():
    """Get all-time battle leaderboard."""
    db = get_db()

    pipeline = [
        {"$unwind": "$leaderboard"},
        {"$group": {
            "_id": "$leaderboard.userId",
            "totalScore": {"$sum": "$leaderboard.score"},
            "totalCorrect": {"$sum": "$leaderboard.correctAnswers"},
            "gamesPlayed": {"$sum": 1},
            "wins": {"$sum": {"$cond": [{"$eq": ["$leaderboard.rank", 1]}, 1, 0]}}
        }},
        {"$sort": {"totalScore": -1}},
        {"$limit": 50}
    ]

    results = await db.battle_history.aggregate(pipeline).to_list(50)
    return {"message": "leaderboard", "payload": results}