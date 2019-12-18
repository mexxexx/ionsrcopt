import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

def main():
    files = {
        './Results/JanNov2016.csv' : {'label' : 0, 'color' : '#880000'},
        './Results/JanNov2016_unstable.csv' : {'label' : 1, 'color' : '#FFAAAA'},
        './Results/JanNov2018.csv' : {'label' : 0, 'color' : '#000088'},
        './Results/JanNov2018_unstable.csv' : {'label' : 1, 'color' : '#AAAAFF'}
    }
    df = read_summaries(files)
    features = ['bias disc', 'gas', 'oven1', 'RF', 'solinj', 'solcen', 'solext', 'HTI', 'BCT25']
    features = [(f, '50%') for f in features]

    X = df[features].values
    y = df['label'].values
    #weights = df[('DURATION', 'in_hours')]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15)

    svc = SVC(C=10.0, kernel='linear', gamma='auto', probability=True)
    svc.fit(X_train, y_train)
    y_pred = svc.predict(X_test)
    print(confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred))
    #print(svc.predict_proba(X))

def read_summaries(files):
    df = None
    for filename, marker in files.items():
        df_new = pd.read_csv(filename, index_col=0, header=[0,1])
        df_new['label'] = marker['label']

        if not df is None:
            df = df.append(df_new, sort=False)
        else:
            df = df_new

    df['label'] = df['label']
    return df[df.index >= 0]

if __name__ == "__main__":
    main()