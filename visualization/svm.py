import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import warnings

from sklearn import svm
from sklearn import metrics
from sklearn import model_selection
from sklearn import preprocessing

def main():
    files = {
        './Results/JanNov2016.csv' : {'label' : 0, 'color' : '#880000'},
        './Results/JanNov2016_unstable.csv' : {'label' : 1, 'color' : '#FFAAAA'},
        './Results/JanNov2018.csv' : {'label' : 0, 'color' : '#000088'},
        './Results/JanNov2018_unstable.csv' : {'label' : 1, 'color' : '#AAAAFF'}
    }
    df = read_summaries(files)
    features = ['bias disc', 'gas', 'RF', 'solinj', 'solcen', 'solext', 'HTI']
    features = [(f, '50%') for f in features]
    feature_ranges = np.array([(df[features[i]].min(), df[features[i]].max()) for i in range(len(features))])

    num_base_points = 80
    shifting_resolution = 2000
    max_deviation = 0.1

    eval_resolution = 10000

    X = df[features].values
    scaler = preprocessing.RobustScaler((10,90)).fit(X)
    X = scaler.transform(X)

    y = df['label'].values
    #weights = df[('DURATION', 'in_hours')]
    kf = model_selection.KFold(n_splits=5, shuffle=True)
    confusion_matrix = np.zeros((2,2))
    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        model = train_model(X_train, y_train)
        confusion_matrix += test_model(model, X_test, y_test)
    
    print(create_classification_report(confusion_matrix))

    model = train_model(X, y, probability=True)

    X = df[features].values
    base_points = X[np.random.permutation(len(X))[:num_base_points]]
    shifted_points = create_shifted_points(base_points, shifting_resolution, max_deviation)
    shifted_points = scale_shifted_points(shifted_points, scaler.transform)

    sensitivities = estimate_feature_sensitivity(model, shifted_points)
    shifted_points = scale_shifted_points(shifted_points, scaler.inverse_transform)

    eval_grid = create_eval_grid(feature_ranges, eval_resolution)
    sensitivities = average_sensitivities(shifted_points, sensitivities, eval_grid, eval_resolution)
    plot_sensitivity(eval_grid, sensitivities, features)

def average_sensitivities(points, sensitivities, eval_grid, eval_resolution):
    dim = len(points)
    num_base_points = len(points[0])
    result = np.empty((dim, eval_resolution))
    result[:] = np.nan

    for d in range(dim):
        evaluations = np.array([np.interp(eval_grid[d], xp=points[d, p, :, d], fp=sensitivities[d, p, :, 0], left=np.nan, right=np.nan) for p in range(num_base_points)])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            result[d] = np.nanmean(evaluations, axis=0)

    return result

def create_eval_grid(feature_ranges, eval_resolution):
    return np.linspace(start=feature_ranges[:,0], stop=feature_ranges[:,1], num=eval_resolution).T

def plot_sensitivity(eval_grid, sensitivities, features):
    dim = len(eval_grid)
    fig, ax = plt.subplots(nrows=dim, ncols=1)
    fig.suptitle('Probability for source to be unstable')
    for d in range(dim):
        ax[d].set_title(features[d])
        ax[d].plot(eval_grid[d], sensitivities[d])

    figManager = plt.get_current_fig_manager()
    figManager.window.showMaximized()
    plt.subplots_adjust(left=0.05, bottom=0.05, right=0.95, top=0.93, wspace=None, hspace=0.4)
    plt.show()

def estimate_feature_sensitivity(model, shifted_points):
    sensitivities = [[model.predict_proba(points) for points in shifted_dim] for shifted_dim in shifted_points]
    return np.array(sensitivities)

def scale_shifted_points(shifted_points, scaling_method):
    shifted_points = [[scaling_method(points) for points in shifted_dim] for shifted_dim in shifted_points]
    return np.array(shifted_points)

def create_shifted_points(base_points, resolution, max_deviation):
    dimension = len(base_points[0])
    
    shifted_points = [[] for d in range(dimension)]

    for d in range(dimension):
        for base_point in base_points:
            points = np.tile(base_point, (resolution, 1))
            min_val = base_point[d] * (1-max_deviation)
            max_val = base_point[d] * (1+max_deviation)
            if (max_val < min_val): 
                swap = min_val
                min_val = max_val
                max_val = swap
            points[:, d] = np.linspace(min_val, max_val, resolution)
            shifted_points[d].append(points)

    return np.array(shifted_points)

def train_model(X, y, probability=False):
    svc = svm.SVC(C=10.0, kernel='rbf', gamma='auto', probability=probability)
    svc.fit(X, y)
    return svc

def test_model(model, X, y):
    y_pred = model.predict(X)
    return metrics.confusion_matrix(y, y_pred)

def create_classification_report(confusion_matrix):
    df = pd.DataFrame()
    df['precision'] = [
        confusion_matrix[0,0] / (confusion_matrix[0,0] + confusion_matrix[1,0]),
        confusion_matrix[1,1] / (confusion_matrix[1,1] + confusion_matrix[0,1])
        ]
    df['recall'] = [
        confusion_matrix[0,0] / (confusion_matrix[0,0] + confusion_matrix[0,1]),
        confusion_matrix[1,1] / (confusion_matrix[1,1] + confusion_matrix[1,0])
        ]
    df['f1-score'] = 2 * df['precision'] * df['recall'] / (df['precision'] + df['recall'])

    return df

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