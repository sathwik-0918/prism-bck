from fastapi import APIRouter
from models.userModel import User
from helpers.userHelper import readUsers, writeUsers, findUserByEmail
import uuid
from database.mongodb import get_db
from api.leaderboardApi import update_leaderboard_points
import httpx

userRouter = APIRouter()            # like exp.Router()

# POST /user — like userApp.post("/user", createUserOrAuthor)
@userRouter.post("/user")
async def createUser(newUser: User):        # req.body → newUser
    try:
        print(f"[API: user] Received user data: {newUser.model_dump()}")
        
        # find user by email — like userAuthor.findOne({email})
        userInDb = findUserByEmail(newUser.email)

        # if user exists
        if userInDb is not None:
            updated = False
            if "userId" not in userInDb:
                userInDb["userId"] = str(uuid.uuid4())
                updated = True
                
            # If user is updating their examTarget from the setup screen
            if newUser.examTarget and newUser.examTarget != "UNKNOWN" and userInDb.get("examTarget") != newUser.examTarget:
                userInDb["examTarget"] = newUser.examTarget
                updated = True

            if updated:
                users = readUsers()
                for i, u in enumerate(users):
                    if u.get("email") == userInDb["email"]:
                        users[i] = userInDb
                        break
                writeUsers(users)
            print(f"[API: user] User exists with role {userInDb['role']}, new role: {newUser.role}")
            
            # ensure user is in leaderboard (points=0 acts as an upsert)
            db = get_db()
            await update_leaderboard_points(userInDb["userId"], "login", 0, db)

            # sync to cloud leaderboard on every login
            try:
                async with httpx.AsyncClient() as client:
                    await client.post(
                        f"http://localhost:8000/api/leaderboard/add-user/{userInDb['userId']}",
                        params={
                            "firstName": userInDb.get("firstName", ""),
                            "profileImageUrl": userInDb.get("profileImageUrl", ""),
                            "examTarget": userInDb.get("examTarget", "")
                        }
                    )
            except Exception:
                pass  # non-blocking
            
            # return FULL existing user including saved examTarget
            return {"message": userInDb["role"], "payload": userInDb}
        
        # new user — like new userAuthor(newUser).save()
        else:
            print(f"[API: user] Creating new user: {newUser.email}")
            users = readUsers()
            userDict = newUser.model_dump()
            userDict["userId"] = str(uuid.uuid4())   # generate unique id
            users.append(userDict)
            writeUsers(users)
            print(f"[API: user] New user created: {newUser.email}")
            
            # add new user to leaderboard
            db = get_db()
            await update_leaderboard_points(userDict["userId"], "login", 0, db)

            # add to cloud leaderboard
            try:
                async with httpx.AsyncClient() as client:
                    await client.post(
                        f"http://localhost:8000/api/leaderboard/add-user/{userDict['userId']}",
                        params={
                            "firstName": userDict.get("firstName", ""),
                            "profileImageUrl": userDict.get("profileImageUrl", ""),
                            "examTarget": userDict.get("examTarget", "")
                        }
                    )
            except Exception:
                pass
            
            return {"message": userDict["role"], "payload": userDict}
    except Exception as e:
        print(f"[API: user] Error: {str(e)}")
        raise



# GET /users — read all users
@userRouter.get("/users")
async def getUsers():
    users = readUsers()
    return {"message": "users", "payload": users}