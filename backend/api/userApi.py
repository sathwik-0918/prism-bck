from fastapi import APIRouter
from models.userModel import User
from helpers.userHelper import readUsers, writeUsers, findUserByEmail
import uuid

userRouter = APIRouter()            # like exp.Router()

# POST /user — like userApp.post("/user", createUserOrAuthor)
@userRouter.post("/user")
async def createUser(newUser: User):        # req.body → newUser
    
    # find user by email — like userAuthor.findOne({email})
    userInDb = findUserByEmail(newUser.email)

    # if user exists
    if userInDb is not None:
        if newUser.role == userInDb["role"]:
            return {"message": newUser.role, "payload": userInDb}
        else:
            return {"message": "Invalid Role"}
    
    # new user — like new userAuthor(newUser).save()
    else:
        users = readUsers()
        userDict = newUser.model_dump()
        userDict["userId"] = str(uuid.uuid4())   # generate unique id
        users.append(userDict)
        writeUsers(users)
        return {"message": userDict["role"], "payload": userDict}


# GET /users — read all users
@userRouter.get("/users")
async def getUsers():
    users = readUsers()
    return {"message": "users", "payload": users}