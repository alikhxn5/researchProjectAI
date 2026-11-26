import pandas as pd

def avg():
    df = pd.read_csv("/Users/justinho/Desktop/Artificial Intellegence/PycharmProjects/researchProjectAI/results/2026-11-26_12-45/results.csv")
    df = df.groupby(["classifier","subset"], as_index = False)[["accuracy","precision","recall","f1"]].mean()

    df.to_csv("/Users/justinho/Desktop/Artificial Intellegence/PycharmProjects/researchProjectAI/results/2026-11-26_12-45/avg.csv")


def main():
    avg()

if __name__ == '__main__':
    main()