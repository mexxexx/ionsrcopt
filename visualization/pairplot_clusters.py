import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def main():
    files = {
        "./Results/MayDec2015.csv": {"label": "2015_stable", "color": "#880000"},
        "./Results/MayDec2015_unstable.csv": {
            "label": "2015_unstable",
            "color": "#FFAAAA",
        },
        "./Results/JanNov2016.csv": {"label": "2016_stable", "color": "#880000"},
        "./Results/JanNov2016_unstable.csv": {
            "label": "2016_unstable",
            "color": "#FFAAAA",
        },
        #'./Results/JanNov2018.csv' : {'label' : '2018_stable', 'color' : '#000088'},
        #'./Results/JanNov2018_unstable.csv' : {'label' : '2018_unstable', 'color' : '#AAAAFF'}
    }
    df = read_summaries(files)
    features = [
        "bias disc",
        "gas",
        "oven1",
        "RF",
        "solinj",
        "solcen",
        "solext",
        "HTI",
        "BCT25",
    ]
    features = [(f, "50%") for f in features]

    sns.pairplot(df, vars=features, hue="label")
    plt.show()


def read_summaries(files):
    df = None
    for filename, marker in files.items():
        df_new = pd.read_csv(filename, index_col=0, header=[0, 1])
        df_new["label"] = marker["label"]

        if not df is None:
            df = df.append(df_new, sort=False)
        else:
            df = df_new

    df["label"] = df["label"].astype("category")
    return df


if __name__ == "__main__":
    main()
