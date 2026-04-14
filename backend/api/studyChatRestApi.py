# api/studyChatRestApi.py
# REST API endpoints for study chat (non-real-time operations)
# friend requests, group management, message history, file serving

from fastapi import APIRouter, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel
from typing import List, Optional
from database.mongodb import get_chat_db as get_db
from datetime import datetime
import uuid

studyChatRouter = APIRouter()


def now():
    return datetime.utcnow().isoformat() + "Z"


# ── USER PROFILE ──────────────────────────────────────────────────────────

class UpdateChatProfileRequest(BaseModel):
    userId: str
    displayName: str
    avatar: Optional[str] = ""
    bio: Optional[str] = ""
    studyGoal: Optional[str] = ""


@studyChatRouter.post("/studychat/profile")
async def updateChatProfile(req: UpdateChatProfileRequest):
    """Create or update chat profile."""
    db = get_db()
    await db.studychat_users.update_one(
        {"userId": req.userId},
        {"$set": {
            "userId": req.userId,
            "displayName": req.displayName,
            "avatar": req.avatar,
            "bio": req.bio,
            "studyGoal": req.studyGoal,
            "updatedAt": now()
        }},
        upsert=True
    )
    return {"message": "profile updated"}


@studyChatRouter.get("/studychat/profile/{userId}")
async def getChatProfile(userId: str):
    db = get_db()
    profile = await db.studychat_users.find_one({"userId": userId}, {"_id": 0, "data": 0})
    return {"message": "profile", "payload": profile}


# ── FRIEND SYSTEM ─────────────────────────────────────────────────────────

class FriendRequestBody(BaseModel):
    fromUserId: str
    toUserId: str


@studyChatRouter.post("/studychat/friend-request")
async def sendFriendRequest(req: FriendRequestBody):
    """Send a friend request."""
    db = get_db()

    if req.fromUserId == req.toUserId:
        return {"message": "cannot add yourself"}

    # check if already friends or request pending
    existing = await db.studychat_friendrequests.find_one({
        "$or": [
            {"fromUserId": req.fromUserId, "toUserId": req.toUserId},
            {"fromUserId": req.toUserId, "toUserId": req.fromUserId}
        ]
    })

    if existing:
        return {"message": "request already exists", "payload": existing}

    # check if already friends
    already_friends = await db.studychat_friends.find_one({
        "users": {"$all": [req.fromUserId, req.toUserId]}
    })
    if already_friends:
        return {"message": "already friends"}

    request_doc = {
        "requestId": str(uuid.uuid4()),
        "fromUserId": req.fromUserId,
        "toUserId": req.toUserId,
        "status": "pending",
        "createdAt": now()
    }
    await db.studychat_friendrequests.insert_one(request_doc)
    request_doc.pop("_id", None)

    print(f"[StudyChat] Friend request: {req.fromUserId} → {req.toUserId}")
    return {"message": "request sent", "payload": request_doc}


class RespondFriendRequest(BaseModel):
    requestId: str
    userId: str
    accept: bool


@studyChatRouter.post("/studychat/friend-request/respond")
async def respondFriendRequest(req: RespondFriendRequest):
    """Accept or reject a friend request."""
    db = get_db()

    request = await db.studychat_friendrequests.find_one(
        {"requestId": req.requestId, "toUserId": req.userId}
    )
    if not request:
        raise HTTPException(404, "Request not found")

    if req.accept:
        # create friendship
        await db.studychat_friends.insert_one({
            "friendshipId": str(uuid.uuid4()),
            "users": sorted([request["fromUserId"], request["toUserId"]]),
            "createdAt": now()
        })
        await db.studychat_friendrequests.update_one(
            {"requestId": req.requestId},
            {"$set": {"status": "accepted"}}
        )
        return {"message": "accepted"}
    else:
        await db.studychat_friendrequests.update_one(
            {"requestId": req.requestId},
            {"$set": {"status": "rejected"}}
        )
        return {"message": "rejected"}


@studyChatRouter.get("/studychat/friends/{userId}")
async def getFriends(userId: str):
    """Get all friends of a user."""
    db = get_db()

    friendships = await db.studychat_friends.find(
        {"users": userId}, {"_id": 0}
    ).to_list(200)

    friend_ids = []
    for f in friendships:
        other = [u for u in f["users"] if u != userId]
        if other:
            friend_ids.append(other[0])

    # get friend profiles
    friends = []
    for fid in friend_ids:
        profile = await db.studychat_users.find_one(
            {"userId": fid}, {"_id": 0, "data": 0}
        )
        if profile:
            friends.append(profile)

    return {"message": "friends", "payload": friends}


@studyChatRouter.get("/studychat/friend-requests/{userId}")
async def getPendingRequests(userId: str):
    """Get pending friend requests for a user."""
    db = get_db()
    requests = await db.studychat_friendrequests.find(
        {"toUserId": userId, "status": "pending"},
        {"_id": 0}
    ).to_list(50)
    return {"message": "requests", "payload": requests}


@studyChatRouter.delete("/studychat/friends/{userId}/{friendId}")
async def removeFriend(userId: str, friendId: str):
    """Remove a friend."""
    db = get_db()
    await db.studychat_friends.delete_one(
        {"users": {"$all": [userId, friendId]}}
    )
    return {"message": "removed"}


@studyChatRouter.post("/studychat/block/{userId}/{blockId}")
async def blockUser(userId: str, blockId: str):
    """Block a user."""
    db = get_db()
    await db.studychat_blocked.update_one(
        {"userId": userId},
        {"$addToSet": {"blockedUsers": blockId}},
        upsert=True
    )
    return {"message": "blocked"}


# ── CONVERSATIONS ─────────────────────────────────────────────────────────

@studyChatRouter.get("/studychat/conversations/{userId}")
async def getConversations(userId: str):
    """Get all DM conversations for a user, sorted by last message."""
    db = get_db()

    convos = await db.studychat_conversations.find(
        {"participants": userId},
        {"_id": 0}
    ).sort("lastMessageTime", -1).to_list(100)

    # enrich with other user's profile
    enriched = []
    for c in convos:
        other_id = next((p for p in c["participants"] if p != userId), None)
        if other_id:
            profile = await db.studychat_users.find_one(
                {"userId": other_id}, {"_id": 0, "data": 0}
            )
            c["otherUser"] = profile or {"userId": other_id}
        enriched.append(c)

    return {"message": "conversations", "payload": enriched}


@studyChatRouter.get("/studychat/messages/dm/{userId}/{otherUserId}")
async def getDMMessages(userId: str, otherUserId: str, skip: int = 0, limit: int = 50):
    """Get DM message history with pagination."""
    db = get_db()
    convo_id = "_".join(sorted([userId, otherUserId]))

    messages = await db.studychat_messages.find(
        {
            "conversationId": convo_id,
            "deletedFor": {"$ne": userId},
            "isDeleted": False
        },
        {"_id": 0, "data": 0}
    ).sort("timestamp", -1).skip(skip).limit(limit).to_list(limit)

    # reverse for chronological order
    messages.reverse()

    # enrich with sender info
    for msg in messages:
        sender = await db.studychat_users.find_one(
            {"userId": msg["fromUserId"]}, {"_id": 0, "data": 0}
        )
        msg["senderInfo"] = sender or {"userId": msg["fromUserId"]}

    return {"message": "messages", "payload": messages}


@studyChatRouter.get("/studychat/messages/group/{groupId}")
async def getGroupMessages(groupId: str, userId: str, skip: int = 0, limit: int = 50):
    """Get group message history."""
    db = get_db()

    # verify membership
    group = await db.studychat_groups.find_one(
        {"groupId": groupId, "members.userId": userId}
    )
    if not group:
        raise HTTPException(403, "Not a member")

    messages = await db.studychat_messages.find(
        {
            "groupId": groupId,
            "deletedFor": {"$ne": userId},
            "isDeleted": False
        },
        {"_id": 0, "data": 0}
    ).sort("timestamp", -1).skip(skip).limit(limit).to_list(limit)

    messages.reverse()

    for msg in messages:
        sender = await db.studychat_users.find_one(
            {"userId": msg["fromUserId"]}, {"_id": 0, "data": 0}
        )
        msg["senderInfo"] = sender or {"userId": msg["fromUserId"]}

    return {"message": "messages", "payload": messages}


@studyChatRouter.get("/studychat/messages/pinned/{groupId}")
async def getPinnedMessages(groupId: str):
    """Get pinned messages in a group."""
    db = get_db()
    pinned = await db.studychat_messages.find(
        {"groupId": groupId, "isPinned": True},
        {"_id": 0, "data": 0}
    ).to_list(20)
    return {"message": "pinned", "payload": pinned}


@studyChatRouter.get("/studychat/search/messages/{userId}")
async def searchMessages(userId: str, q: str):
    """Search messages across all conversations."""
    db = get_db()
    results = await db.studychat_messages.find(
        {
            "$or": [
                {"conversationId": {"$regex": userId}},
                {"groupId": {"$exists": True}}
            ],
            "content": {"$regex": q, "$options": "i"},
            "isDeleted": False
        },
        {"_id": 0, "data": 0}
    ).limit(30).to_list(30)
    return {"message": "results", "payload": results}


# ── GROUPS ────────────────────────────────────────────────────────────────

class CreateGroupRequest(BaseModel):
    creatorId: str
    name: str
    description: Optional[str] = ""
    subject: Optional[str] = ""    # Physics, Chemistry etc
    memberIds: List[str] = []


@studyChatRouter.post("/studychat/groups")
async def createGroup(req: CreateGroupRequest):
    """Create a study group."""
    db = get_db()

    group_id = str(uuid.uuid4())
    members = [{"userId": req.creatorId, "role": "owner", "joinedAt": now()}]

    for mid in req.memberIds:
        if mid != req.creatorId:
            members.append({"userId": mid, "role": "member", "joinedAt": now()})

    group = {
        "groupId": group_id,
        "name": req.name,
        "description": req.description,
        "subject": req.subject,
        "creatorId": req.creatorId,
        "members": members,
        "lastMessage": "",
        "lastMessageTime": now(),
        "createdAt": now()
    }

    await db.studychat_groups.insert_one(group)
    group.pop("_id", None)

    print(f"[StudyChat] Group created: '{req.name}' by {req.creatorId}")
    return {"message": "group created", "payload": group}


@studyChatRouter.get("/studychat/groups/{userId}")
async def getUserGroups(userId: str):
    """Get all groups a user belongs to."""
    db = get_db()
    groups = await db.studychat_groups.find(
        {"members.userId": userId},
        {"_id": 0, "data": 0}
    ).sort("lastMessageTime", -1).to_list(50)
    return {"message": "groups", "payload": groups}


class AddMemberRequest(BaseModel):
    groupId: str
    adminId: str
    newMemberId: str


@studyChatRouter.post("/studychat/groups/add-member")
async def addMember(req: AddMemberRequest):
    db = get_db()
    group = await db.studychat_groups.find_one({"groupId": req.groupId})
    if not group:
        raise HTTPException(404, "Group not found")

    # check admin
    member = next((m for m in group["members"] if m["userId"] == req.adminId), None)
    if not member or member["role"] not in ["admin", "owner"]:
        raise HTTPException(403, "Not admin")

    # add member
    await db.studychat_groups.update_one(
        {"groupId": req.groupId},
        {"$push": {"members": {
            "userId": req.newMemberId,
            "role": "member",
            "joinedAt": now()
        }}}
    )
    return {"message": "member added"}


@studyChatRouter.delete("/studychat/groups/{groupId}/member/{memberId}")
async def removeMember(groupId: str, memberId: str, adminId: str):
    db = get_db()
    group = await db.studychat_groups.find_one({"groupId": groupId})
    if not group:
        raise HTTPException(404)

    admin = next((m for m in group["members"] if m["userId"] == adminId), None)
    if not admin or admin["role"] not in ["admin", "owner"]:
        raise HTTPException(403)

    await db.studychat_groups.update_one(
        {"groupId": groupId},
        {"$pull": {"members": {"userId": memberId}}}
    )
    return {"message": "removed"}


# ── FILE SERVING ──────────────────────────────────────────────────────────

@studyChatRouter.get("/studychat/file/{fileId}")
async def serveFile(fileId: str):
    """Serve uploaded file by ID."""
    db = get_db()
    file_doc = await db.studychat_files.find_one({"fileId": fileId})
    if not file_doc:
        raise HTTPException(404)

    content_type_map = {
        "image": "image/jpeg",
        "voice": "audio/webm",
        "file": "application/octet-stream"
    }
    content_type = content_type_map.get(file_doc["fileType"], "application/octet-stream")

    return Response(
        content=file_doc["data"],
        media_type=content_type,
        headers={"Content-Disposition": f"inline; filename={file_doc['fileName']}"}
    )


# ── SEARCH USERS ─────────────────────────────────────────────────────────

@studyChatRouter.get("/studychat/search/users")
async def searchUsers(q: str, currentUserId: str = ""):
    """Search users by name for friend requests."""
    from helpers.userHelper import readUsers
    all_users = readUsers()
    q_lower = q.lower()
    
    results = []
    for u in all_users:
        user_id = u.get("userId")
        if not user_id or user_id == currentUserId:
            continue
            
        full_name = f"{u.get('firstName', '')} {u.get('lastName', '')}".strip()
        if q_lower in full_name.lower():
            results.append({
                "userId": user_id,
                "displayName": full_name,
                "avatar": u.get("profileImageUrl", ""),
                "studyGoal": u.get("examTarget", "")
            })
            if len(results) >= 20:
                break

    return {"message": "users", "payload": results}