import pandas as pd


def main():
    bots = compileText(readCSV("Bots.csv"))
    users = compileText(readCSV("RandomUser.csv"))
    print(bots.head())
    print(users.head())


def readCSV(file):
    df = pd.read_csv(file)
    return df

# 0 is user 1 is bot
def compileText(dataframe):

    classification = []
    if dataframe.iloc[0,3] == 'Bot':
        for i in range(len(dataframe)):
            classification.append(1)
    else:
        for i in range(len(dataframe)):
            classification.append(0)
    combined = dataframe['description'] + " " + dataframe['first_commit_message'] + " " + dataframe['last_commit_message']
    compiled = pd.DataFrame ({
        "First_Author_Type": dataframe['first_commit_author_type'],
        "Last_Author_Type": dataframe['last_commit_author_type'],
        "Commit_Text" : combined,
        "Author":classification
    })
    return compiled

if __name__ == "__main__":
    main()
