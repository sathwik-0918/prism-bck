# api/studyChatApi.py
# Real-time study chat backend using Socket.IO + FastAPI
# Features: DMs, group chats, typing indicators, online status,
# message reactions, replies, read receipts, file sharing, friend system
# All data stored in MongoDB — fully persistent

import socketio
from datetime import datetime, timedelta
from typing import Optional
import uuid
import base64

# ── Socket.IO server ─────────────────────────────────────────────────────
# async_mode='asgi' integrates perfectly with FastAPI/uvicorn
sio = socketio.AsyncServer(
    async_mode='asgi',
    cors_allowed_origins='*',
    ping_timeout=60,
    ping_interval=25,
    max_http_buffer_size=10 * 1024 * 1024  # 10MB for file transfers
)


def now() -> str:
    return datetime.utcnow().isoformat()


# ── In-memory presence tracking ───────────────────────────────────────────
# Maps userId → set of socketIds (user can have multiple tabs)
online_users: dict[str, set] = {}
# Maps socketId → userId
socket_to_user: dict[str, str] = {}


def get_online_status(user_id: str) -> str:
    return "online" if user_id in online_users and online_users[user_id] else "offline"


async def broadcast_presence(sio_instance, user_id: str, status: str):
    """Notify all users of presence change."""
    await sio_instance.emit("user_presence", {
        "userId": user_id,
        "status": status,
        "lastSeen": now()
    })


# ════════════════════════════════════════════════════════════════════════════
# CONNECTION EVENTS
# ════════════════════════════════════════════════════════════════════════════

@sio.event
async def connect(sid, environ, auth):
    """
    Client connected. auth contains userId from Clerk.
    Joins user to their personal room and all their group rooms.
    """
    user_id = auth.get("userId") if auth else None
    if not user_id:
        print(f"[Socket] Rejected unauthenticated connection: {sid}")
        return False  # reject connection

    # track presence
    socket_to_user[sid] = user_id
    if user_id not in online_users:
        online_users[user_id] = set()
    online_users[user_id].add(sid)

    # join personal room (for DMs and notifications)
    await sio.enter_room(sid, f"user_{user_id}")

    # load and join all group rooms this user belongs to
    from database.mongodb import get_db
    db = get_db()
    groups = await db.studychat_groups.find(
        {"members.userId": user_id},
        {"groupId": 1}
    ).to_list(100)
    for group in groups:
        await sio.enter_room(sid, f"group_{group['groupId']}")

    # broadcast online status
    await broadcast_presence(sio, user_id, "online")

    # update last seen in DB
    await db.studychat_users.update_one(
        {"userId": user_id},
        {"$set": {"lastSeen": now(), "status": "online"}},
        upsert=True
    )

    print(f"[Socket] Connected: {user_id} (sid={sid})")


@sio.event
async def disconnect(sid):
    """Client disconnected. Update presence."""
    user_id = socket_to_user.pop(sid, None)
    if not user_id:
        return

    online_users.get(user_id, set()).discard(sid)

    # if no more connections for this user → offline
    if not online_users.get(user_id):
        online_users.pop(user_id, None)
        await broadcast_presence(sio, user_id, "offline")

        # update DB
        from database.mongodb import get_db
        db = get_db()
        await db.studychat_users.update_one(
            {"userId": user_id},
            {"$set": {"lastSeen": now(), "status": "offline"}}
        )

    print(f"[Socket] Disconnected: {user_id} (sid={sid})")


# ════════════════════════════════════════════════════════════════════════════
# DIRECT MESSAGES
# ════════════════════════════════════════════════════════════════════════════

@sio.event
async def send_dm(sid, data):
    """
    Send a direct message.
    data: {toUserId, content, type, replyTo, fileData, fileName}
    type: 'text' | 'image' | 'file' | 'voice'
    """
    from_user = socket_to_user.get(sid)
    if not from_user:
        return

    to_user = data.get("toUserId")
    if not to_user:
        return

    from database.mongodb import get_db
    db = get_db()

    # build conversation ID (sorted so A↔B == B↔A)
    convo_id = "_".join(sorted([from_user, to_user]))

    # handle file
    file_url = None
    if data.get("fileData") and data.get("type") in ["image", "file", "voice"]:
        file_url = await save_file(db, data["fileData"], data.get("fileName", "file"),
                                   data.get("type", "file"))

    message = {
        "messageId": str(uuid.uuid4()),
        "conversationId": convo_id,
        "fromUserId": from_user,
        "toUserId": to_user,
        "content": data.get("content", ""),
        "type": data.get("type", "text"),
        "fileUrl": file_url,
        "fileName": data.get("fileName", ""),
        "replyTo": data.get("replyTo"),   # messageId being replied to
        "reactions": {},
        "readBy": [from_user],
        "edited": False,
        "deletedFor": [],
        "timestamp": now(),
        "isDeleted": False,
    }

    await db.studychat_messages.insert_one(message)
    message.pop("_id", None)

    # get sender info for display
    sender = await db.studychat_users.find_one(
        {"userId": from_user}, {"_id": 0}
    )
    message["senderInfo"] = sender or {"userId": from_user}

    # deliver to recipient's personal room + sender
    await sio.emit("new_dm", message, room=f"user_{to_user}")
    await sio.emit("new_dm", message, room=f"user_{from_user}")

    # update conversation last message
    await db.studychat_conversations.update_one(
        {"conversationId": convo_id},
        {
            "$set": {
                "lastMessage": message["content"] or f"[{message['type']}]",
                "lastMessageTime": now(),
                "lastMessageFrom": from_user,
                "participants": sorted([from_user, to_user])
            },
            "$inc": {f"unreadCount.{to_user}": 1}
        },
        upsert=True
    )

    print(f"[Socket] DM: {from_user} → {to_user}: '{str(message['content'])[:40]}'")


@sio.event
async def typing_dm(sid, data):
    """Broadcast typing indicator for DM."""
    from_user = socket_to_user.get(sid)
    to_user = data.get("toUserId")
    if not from_user or not to_user:
        return
    await sio.emit("typing_dm", {
        "fromUserId": from_user,
        "isTyping": data.get("isTyping", True)
    }, room=f"user_{to_user}")


@sio.event
async def mark_read_dm(sid, data):
    """Mark all messages in a DM conversation as read."""
    user_id = socket_to_user.get(sid)
    other_user = data.get("otherUserId")
    if not user_id or not other_user:
        return

    from database.mongodb import get_db
    db = get_db()

    convo_id = "_".join(sorted([user_id, other_user]))

    # mark messages as read
    await db.studychat_messages.update_many(
        {
            "conversationId": convo_id,
            "fromUserId": other_user,
            "readBy": {"$ne": user_id}
        },
        {"$addToSet": {"readBy": user_id}}
    )

    # reset unread count
    await db.studychat_conversations.update_one(
        {"conversationId": convo_id},
        {"$set": {f"unreadCount.{user_id}": 0}}
    )

    # notify sender their messages were read
    await sio.emit("messages_read", {
        "conversationId": convo_id,
        "readBy": user_id
    }, room=f"user_{other_user}")


@sio.event
async def react_to_message(sid, data):
    """Add/remove reaction to any message."""
    user_id = socket_to_user.get(sid)
    if not user_id:
        return

    from database.mongodb import get_db
    db = get_db()

    message_id = data.get("messageId")
    emoji = data.get("emoji")
    room_type = data.get("roomType", "dm")  # dm | group

    msg = await db.studychat_messages.find_one({"messageId": message_id})
    if not msg:
        return

    reactions = msg.get("reactions", {})
    if emoji not in reactions:
        reactions[emoji] = []

    if user_id in reactions[emoji]:
        reactions[emoji].remove(user_id)  # toggle off
    else:
        reactions[emoji].append(user_id)  # toggle on

    if not reactions[emoji]:
        del reactions[emoji]

    await db.studychat_messages.update_one(
        {"messageId": message_id},
        {"$set": {"reactions": reactions}}
    )

    reaction_update = {"messageId": message_id, "reactions": reactions}

    if room_type == "dm":
        convo_id = msg.get("conversationId", "")
        parts = convo_id.split("_")
        for p in parts:
            await sio.emit("reaction_updated", reaction_update, room=f"user_{p}")
    else:
        group_id = msg.get("groupId")
        await sio.emit("reaction_updated", reaction_update, room=f"group_{group_id}")


@sio.event
async def edit_message(sid, data):
    """Edit a sent message."""
    user_id = socket_to_user.get(sid)
    message_id = data.get("messageId")
    new_content = data.get("content", "")

    if not user_id or not message_id:
        return

    from database.mongodb import get_db
    db = get_db()

    msg = await db.studychat_messages.find_one({"messageId": message_id})
    if not msg or msg.get("fromUserId") != user_id:
        return  # can only edit own messages

    await db.studychat_messages.update_one(
        {"messageId": message_id},
        {"$set": {"content": new_content, "edited": True, "editedAt": now()}}
    )

    update = {"messageId": message_id, "content": new_content, "edited": True}

    if msg.get("groupId"):
        await sio.emit("message_edited", update, room=f"group_{msg['groupId']}")
    else:
        convo_id = msg.get("conversationId", "")
        for p in convo_id.split("_"):
            await sio.emit("message_edited", update, room=f"user_{p}")


@sio.event
async def delete_message(sid, data):
    """Delete message — for everyone or just for me."""
    user_id = socket_to_user.get(sid)
    message_id = data.get("messageId")
    delete_for_everyone = data.get("deleteForEveryone", False)

    if not user_id or not message_id:
        return

    from database.mongodb import get_db
    db = get_db()

    msg = await db.studychat_messages.find_one({"messageId": message_id})
    if not msg or msg.get("fromUserId") != user_id:
        return

    if delete_for_everyone:
        await db.studychat_messages.update_one(
            {"messageId": message_id},
            {"$set": {"isDeleted": True, "content": "This message was deleted"}}
        )
        update = {"messageId": message_id, "isDeleted": True, "deleteForEveryone": True}
        if msg.get("groupId"):
            await sio.emit("message_deleted", update, room=f"group_{msg['groupId']}")
        else:
            for p in msg.get("conversationId", "").split("_"):
                await sio.emit("message_deleted", update, room=f"user_{p}")
    else:
        # delete for me only
        await db.studychat_messages.update_one(
            {"messageId": message_id},
            {"$addToSet": {"deletedFor": user_id}}
        )


# ════════════════════════════════════════════════════════════════════════════
# GROUP CHAT
# ════════════════════════════════════════════════════════════════════════════

@sio.event
async def send_group_message(sid, data):
    """Send message to a group."""
    user_id = socket_to_user.get(sid)
    group_id = data.get("groupId")
    if not user_id or not group_id:
        return

    from database.mongodb import get_db
    db = get_db()

    # verify membership
    group = await db.studychat_groups.find_one(
        {"groupId": group_id, "members.userId": user_id}
    )
    if not group:
        return

    file_url = None
    if data.get("fileData") and data.get("type") in ["image", "file", "voice"]:
        file_url = await save_file(db, data["fileData"], data.get("fileName", ""),
                                   data.get("type", "file"))

    message = {
        "messageId": str(uuid.uuid4()),
        "groupId": group_id,
        "fromUserId": user_id,
        "content": data.get("content", ""),
        "type": data.get("type", "text"),
        "fileUrl": file_url,
        "fileName": data.get("fileName", ""),
        "replyTo": data.get("replyTo"),
        "reactions": {},
        "readBy": [user_id],
        "edited": False,
        "deletedFor": [],
        "isPinned": False,
        "timestamp": now(),
        "isDeleted": False,
    }

    await db.studychat_messages.insert_one(message)
    message.pop("_id", None)

    sender = await db.studychat_users.find_one({"userId": user_id}, {"_id": 0})
    message["senderInfo"] = sender or {"userId": user_id}

    # broadcast to all group members
    await sio.emit("new_group_message", message, room=f"group_{group_id}")

    # update group last message
    await db.studychat_groups.update_one(
        {"groupId": group_id},
        {"$set": {
            "lastMessage": message["content"] or f"[{message['type']}]",
            "lastMessageTime": now(),
            "lastMessageFrom": user_id
        }}
    )


@sio.event
async def typing_group(sid, data):
    """Broadcast typing in group."""
    user_id = socket_to_user.get(sid)
    group_id = data.get("groupId")
    if not user_id or not group_id:
        return
    await sio.emit("group_typing", {
        "groupId": group_id,
        "userId": user_id,
        "isTyping": data.get("isTyping", True)
    }, room=f"group_{group_id}", skip_sid=sid)


@sio.event
async def pin_message(sid, data):
    """Pin a message in group (admin only)."""
    user_id = socket_to_user.get(sid)
    group_id = data.get("groupId")
    message_id = data.get("messageId")

    if not user_id or not group_id or not message_id:
        return

    from database.mongodb import get_db
    db = get_db()

    # check if admin
    group = await db.studychat_groups.find_one({"groupId": group_id})
    if not group:
        return
    member = next((m for m in group.get("members", []) if m["userId"] == user_id), None)
    if not member or member.get("role") not in ["admin", "owner"]:
        return

    await db.studychat_messages.update_one(
        {"messageId": message_id},
        {"$set": {"isPinned": True}}
    )

    await sio.emit("message_pinned", {"messageId": message_id, "groupId": group_id},
                   room=f"group_{group_id}")


# ════════════════════════════════════════════════════════════════════════════
# FILE HANDLING
# ════════════════════════════════════════════════════════════════════════════

async def save_file(db, file_data_b64: str, file_name: str, file_type: str) -> Optional[str]:
    """
    Saves base64 encoded file to MongoDB and returns a retrieval ID.
    Supports images, documents, voice notes.
    Max 8MB per file.
    """
    try:
        # decode
        if "," in file_data_b64:
            file_data_b64 = file_data_b64.split(",")[1]

        file_bytes = base64.b64decode(file_data_b64)

        if len(file_bytes) > 8 * 1024 * 1024:
            print(f"[Socket] File too large: {len(file_bytes)} bytes")
            return None

        file_id = str(uuid.uuid4())
        await db.studychat_files.insert_one({
            "fileId": file_id,
            "fileName": file_name,
            "fileType": file_type,
            "data": file_bytes,
            "size": len(file_bytes),
            "uploadedAt": now()
        })

        return f"/api/studychat/file/{file_id}"
    except Exception as e:
        print(f"[Socket] File save error: {e}")
        return None