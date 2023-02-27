import pandas as pd

def loader(path):

    df = pd.read_csv(path)
    return list(df["question"]) , list(df["answer"])
