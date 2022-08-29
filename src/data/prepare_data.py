import pandas as pd

from src.data.load_data import load_data_from_mongo
from src.config.config import BASE_STAMP

def prepare_dataset():
    df_users, df_thoughts = load_data_from_mongo()
    df = df_thoughts.merge(df_users, on="author")
    df["unix"] = (df.created_at - pd.Timestamp("1970-01-01")) // pd.Timedelta('1s')
    df["banned"] = df.banned_at.isna().apply(lambda x: not x)

    df["diff"] = df.banned.values.astype(int) - df.blocked.values.astype(int)
    df = df[df["diff"]!=1]

    df = df[df["unix"] > BASE_STAMP][["created_at", "body", "blocked", "banned"]]
    df.dropna(inplace=True)

    print("Dataset has {} samples".format(len(df)))

    return df