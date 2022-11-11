import pandas
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler



def player_regular_season_outliers():
    title = 'player_regular_season'
    # Load dataset
    data_allstar_original = pandas.read_csv(title+'.txt')

    # Exclude  id, name ,surname and league from data set
    data_allstar = data_allstar_original.iloc[:, 4:]

    data_allstar = data_allstar.to_numpy()

    # print data where it is nan or infinity
    print(np.isnan(data_allstar))
    # replace nan and infinity with numbers or we can replace it with the mean
    data_allstar = np.nan_to_num(data_allstar)

    iforest = IsolationForest(n_estimators=100, max_samples='auto',
                              contamination=0.05, max_features=1.0,
                              bootstrap=False, n_jobs=-1, random_state=1)

    # Returns 1 of inliers, -1 for outliers
    pred = iforest.fit_predict(data_allstar)

    # Extract outliers
    outlier_index = np.where(pred == -1)
    outlier_values = data_allstar[outlier_index]

    # Feature scaling
    sc = StandardScaler()
    X_scaled = sc.fit_transform(data_allstar)
    outlier_values_scaled = sc.transform(outlier_values)

    # Apply PCA to reduce the dimensionality
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    outlier_values_pca = pca.transform(outlier_values_scaled)

    # Plot the data
    sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1])
    sns.scatterplot(x=outlier_values_pca[:, 0],
                    y=outlier_values_pca[:, 1], color='r')
    plt.title("Isolation Forest Outlier Detection ("+title+")",
              fontsize=15, pad=15)
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.savefig(title+".png", dpi=80)

    print('Players Play Off Career')
    f = open(title+"-output.txt", "a")
    for x in outlier_index:
        print(data_allstar_original.iloc[x, :])
        f.write(data_allstar_original.iloc[x, :].to_string())
def player_regular_season_career_outliers():
    title = 'player_regular_season_career'
    # Load dataset
    data_allstar_original = pandas.read_csv(title+'.txt')

    # Exclude  id, name ,surname and league from data set
    data_allstar = data_allstar_original.iloc[:, 4:]

    data_allstar = data_allstar.to_numpy()

    # print data where it is nan or infinity
    print(np.isnan(data_allstar))
    # replace nan and infinity with numbers or we can replace it with the mean ?
    data_allstar = np.nan_to_num(data_allstar)

    iforest = IsolationForest(n_estimators=100, max_samples='auto',
                              contamination=0.05, max_features=1.0,
                              bootstrap=False, n_jobs=-1, random_state=1)

    # Returns 1 of inliers, -1 for outliers
    pred = iforest.fit_predict(data_allstar)

    # Extract outliers
    outlier_index = np.where(pred == -1)
    outlier_values = data_allstar[outlier_index]

    # Feature scaling
    sc = StandardScaler()
    X_scaled = sc.fit_transform(data_allstar)
    outlier_values_scaled = sc.transform(outlier_values)

    # Apply PCA to reduce the dimensionality
    # Normalize and fit the metrics to a PCA to reduce the number of dimensions
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    outlier_values_pca = pca.transform(outlier_values_scaled)

    # Plot the data
    sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1])
    sns.scatterplot(x=outlier_values_pca[:, 0],
                    y=outlier_values_pca[:, 1], color='r')
    plt.title("Isolation Forest Outlier Detection ("+title+")",
              fontsize=15, pad=15)
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.savefig(title+".png", dpi=80)

    print('Players Play Off Career')
    f = open(title+"-output.txt", "a")
    for x in outlier_index:
        print(data_allstar_original.iloc[x, :])
        f.write(data_allstar_original.iloc[x, :].to_string())
def player_playoffs_outliers():
    title = 'player_playoffs'
    # Load dataset
    data_allstar_original = pandas.read_csv(title+'.txt')

    # Exclude  id, name ,surname and league from data set
    data_allstar = data_allstar_original.iloc[:, 4:]

    data_allstar = data_allstar.to_numpy()

    # print data where it is nan or infinity
    print(np.isnan(data_allstar))
    # replace nan and infinity with numbers or we can replace it with the mean ?
    data_allstar = np.nan_to_num(data_allstar)

    iforest = IsolationForest(n_estimators=100, max_samples='auto',
                              contamination=0.05, max_features=1.0,
                              bootstrap=False, n_jobs=-1, random_state=1)

    # Returns 1 of inliers, -1 for outliers
    pred = iforest.fit_predict(data_allstar)

    # Extract outliers
    outlier_index = np.where(pred == -1)
    outlier_values = data_allstar[outlier_index]

    # Feature scaling
    sc = StandardScaler()
    X_scaled = sc.fit_transform(data_allstar)
    outlier_values_scaled = sc.transform(outlier_values)

    # Apply PCA to reduce the dimensionality
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    outlier_values_pca = pca.transform(outlier_values_scaled)

    # Plot the data
    sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1])
    sns.scatterplot(x=outlier_values_pca[:, 0],
                    y=outlier_values_pca[:, 1], color='r')
    plt.title("Isolation Forest Outlier Detection ("+title+")",
              fontsize=15, pad=15)
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.savefig(title+".png", dpi=80)

    print('Players Play Off Career')
    f = open(title+"-output.txt", "a")
    for x in outlier_index:
        print(data_allstar_original.iloc[x, :])
        f.write(data_allstar_original.iloc[x, :].to_string())
def player_playoffs_career_outliers():
    title = 'player_playoffs_career'
    # Load dataset
    data_allstar_original = pandas.read_csv(title+'.txt')

    # Exclude  id, name ,surname and league from data set
    data_allstar = data_allstar_original.iloc[:, 4:]

    data_allstar = data_allstar.to_numpy()

    # print data where it is nan or infinity
    print(np.isnan(data_allstar))
    # replace nan and infinity with numbers or we can replace it with the mean ?
    data_allstar = np.nan_to_num(data_allstar)

    iforest = IsolationForest(n_estimators=100, max_samples='auto',
                              contamination=0.05, max_features=1.0,
                              bootstrap=False, n_jobs=-1, random_state=1)

    # Returns 1 of inliers, -1 for outliers
    pred = iforest.fit_predict(data_allstar)

    # Extract outliers
    outlier_index = np.where(pred == -1)
    outlier_values = data_allstar[outlier_index]

    # Feature scaling
    sc = StandardScaler()
    X_scaled = sc.fit_transform(data_allstar)
    outlier_values_scaled = sc.transform(outlier_values)

    # Apply PCA to reduce the dimensionality
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    outlier_values_pca = pca.transform(outlier_values_scaled)

    # Plot the data
    sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1])
    sns.scatterplot(x=outlier_values_pca[:, 0],
                    y=outlier_values_pca[:, 1], color='r')
    plt.title("Isolation Forest Outlier Detection ("+title+")",
              fontsize=15, pad=15)
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.savefig(title+".png", dpi=80)

    print('Players Play Off Career')
    f = open(title+"-output.txt", "a")
    for x in outlier_index:
        print(data_allstar_original.iloc[x, :])
        f.write(data_allstar_original.iloc[x, :].to_string())

def player_allstar_outliers():
    title = 'player_allstar'
    # Load dataset
    data_allstar_original = pandas.read_csv(title+'.txt')

    # Exclude  id, name ,surname and league from data set
    data_allstar = data_allstar_original.iloc[:, 6:]


    data_allstar = data_allstar.to_numpy()


    # print data where it is nan or infinity
    print(np.isnan(data_allstar))
    # replace nan and infinity with numbers or we can replace it with the mean ?
    data_allstar = np.nan_to_num(data_allstar)

    iforest = IsolationForest(n_estimators=100, max_samples='auto',
                              contamination=0.05, max_features=1.0,
                              bootstrap=False, n_jobs=-1, random_state=1)

    # Returns 1 of inliers, -1 for outliers
    pred = iforest.fit_predict(data_allstar)

    # Extract outliers
    outlier_index = np.where(pred == -1)
    outlier_values = data_allstar[outlier_index]

    # Feature scaling
    sc = StandardScaler()
    X_scaled = sc.fit_transform(data_allstar)
    outlier_values_scaled = sc.transform(outlier_values)

    # Apply PCA to reduce the dimensionality
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    outlier_values_pca = pca.transform(outlier_values_scaled)

    # Plot the data
    sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1])
    sns.scatterplot(x=outlier_values_pca[:, 0],
                    y=outlier_values_pca[:, 1], color='r')
    plt.title("Isolation Forest Outlier Detection ("+title+")",
              fontsize=15, pad=15)
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.savefig(title+".png", dpi=80)

    print('Players Play Off Career')
    f = open(title+"-output.txt", "a")
    for x in outlier_index:
        print(data_allstar_original.iloc[x, :])
        f.write(data_allstar_original.iloc[x, :].to_string())

