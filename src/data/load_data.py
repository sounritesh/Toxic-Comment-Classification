from pymongo import MongoClient
from tqdm import tqdm
import pandas as pd

from src.config.secrets import secrets

def load_data_from_mongo():
    client = MongoClient(secrets["client"])

    user_list = []
    users = client.account.users
    for user in tqdm(users.find(no_cursor_timeout=True), total=users.count_documents({})):
        user_list.append(user)
    print("{} users retrieved from client.".format(len(user_list)))
    df_users = pd.DataFrame(user_list)[["_id", "banned_at"]]
    df_users.rename(columns = {"_id": "author"}, inplace=True)

    thought_list = []
    thoughts = client.forum.thoughts
    for thought in tqdm(thoughts.find(no_cursor_timeout=True), total=thoughts.count_documents({})):
        thought_list.append(thought)
    print("{} thoughts retrieved from client.".format(len(thought_list)))
    df_thoughts = pd.DataFrame(thought_list)[["created_at", "author", "body", "blocked"]]

    return df_users, df_thoughts