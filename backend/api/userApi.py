from fastapi import APIRouter
from models.userModel import User
from helpers.userHelper import readUsers, writeUsers, findUserByEmail
import uuid

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
            if "userId" not in userInDb:
                userInDb["userId"] = str(uuid.uuid4())
                users = readUsers()
                for i, u in enumerate(users):
                    if u.get("email") == userInDb["email"]:
                        users[i] = userInDb
                        break
                writeUsers(users)
            print(f"[API: user] User exists with role {userInDb['role']}, new role: {newUser.role}")
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
            return {"message": userDict["role"], "payload": userDict}
    except Exception as e:
        print(f"[API: user] Error: {str(e)}")
        raise



# GET /users — read all users
@userRouter.get("/users")
async def getUsers():
    users = readUsers()
    return {"message": "users", "payload": users}