import json
import os

USERS_FILE = "data/users.json"

# read all users from json file — like MongoDB .find()
def readUsers():
    if not os.path.exists(USERS_FILE):
        return []
    with open(USERS_FILE, "r") as f:
        return json.load(f)

# write users back to json file — like .save()
def writeUsers(users):
    with open(USERS_FILE, "w") as f:
        json.dump(users, f, indent=2)

# find user by email — like .findOne({email})
def findUserByEmail(email):
    users = readUsers()
    for user in users:
        if user["email"] == email:
            return user
    return None