from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

# pca step is described in the report
def PCA(X, num_components):
    N, M = X.shape
    # find mean
    means = X - np.mean(X , axis = 0)

    # calculate covariance matrix
    cov = np.zeros((M, M))
    for i in range(M):
        mean_col_i = np.sum(X[:, i]) / N
        for j in range(M):
            mean_col_j = np.sum(X[:, j]) / N
            cov[i, j] = np.sum((X[:, i] - mean_col_i) * (X[:, j] - mean_col_j)) / (N - 1)

    eigen_values , eigen_vectors = np.linalg.eigh(cov)
    index = np.argsort(eigen_values)[::-1]
    s_eigenvectors = eigen_vectors[:,index]
    eigenvector_dimension = s_eigenvectors[:,0:num_components]
    reduced_dim = np.dot(eigenvector_dimension.transpose() , means.transpose() ).transpose()
    return reduced_dim

# lda step is described in the report
def LDA(X, y, num_components):
    dimensions = X.shape[1]
    classes = np.unique(y)

    mu = np.mean(X, axis=0)
    SW = np.zeros((dimensions, dimensions))
    SB = np.zeros((dimensions, dimensions))
    for label in classes:
        sample = X[y == label]
        sample_mean = np.mean(sample, axis=0)
        SW += (sample - sample_mean).T.dot((sample - sample_mean))

        mean_diff = (sample_mean - mu).values.reshape(dimensions, 1)
        SB += sample.shape[0] * (mean_diff).dot(mean_diff.T)

    eigenvalues, eigenvectors = np.linalg.eig(np.linalg.inv(SW).dot(SB))
    eigenvectors = eigenvectors.T
    index = np.argsort(abs(eigenvalues))[::-1]
    eigenvalues = eigenvalues[index]
    eigenvectors = eigenvectors[index]
    ld = eigenvectors[0 : num_components]

    return np.dot(X, ld.T)
    
# create 1d graph of pca, principal_df is the principal component returned by the pca algorithm
def create1DGraph(ax, principal_df):
    principal_df_copy = pd.DataFrame(principal_df , columns = ['PC1', 'diagnosis'])
    M_df = principal_df_copy[principal_df_copy.diagnosis != 'M']
    B_df = principal_df_copy[principal_df_copy.diagnosis != 'B']
    ax.hist([M_df['PC1'], B_df['PC1']], label=['m', 'b'])
    ax.set_xlabel('Principal Component 1', fontsize=10)
    ax.legend(loc='upper right')

# create 2d graph of pca, principal_df is the principal component returned by the pca algorithm
def create2DGraph(ax, principal_df):
    principal_df_copy = pd.DataFrame(principal_df , columns = ['PC1','PC2', 'diagnosis'])
    ax.set_xlabel('Principal Component 1',fontsize = 10)  
    ax.set_ylabel('Principal Component 2',fontsize = 10)
    targets=['M','B']
    colors=['r','g']
    for target,color in zip(targets,colors):    
        indicesToKeep = principal_df_copy['diagnosis'] == target  
        ax.scatter(principal_df_copy.loc[indicesToKeep,'PC1'],
                principal_df_copy.loc[indicesToKeep,'PC2'],
                c=color,
                s=10)
    ax.legend(targets)  
    ax.grid()

# conver diagnoses to numeric value, 0 Benign or 1 malignant
def diagnose(x):
    if x =='M':
        return 1
    else:
        return 0

# create 3d graph of pca, principal_df is the principal component returned by the pca algorithm
def create3DGraph(ax2, principal_df):
    principal_df_copy = pd.DataFrame(principal_df , columns = ['PC1','PC2', 'PC3', 'diagnosis'])
    df_diag= principal_df_copy['diagnosis'].apply(diagnose)
    ax2.scatter(principal_df_copy['PC1'], principal_df_copy['PC2'], principal_df_copy['PC3'], c=df_diag, s=10)
    ax2.legend(['Malign'])
    ax2.set_xlabel('PC 1')
    ax2.set_ylabel('PC 2')
    ax2.set_zlabel('PC 3')
    ax2.view_init(30, 120)

def use_pca():
    df = pd.read_csv('breast_cancer_data.csv')
    df = df.drop(['id', 'Unnamed: 32'], axis=1)

    # data cleaning
    X = df.drop('diagnosis', axis=1)
    y = df['diagnosis']

    # scale the data
    X_transformed = StandardScaler().fit_transform(X)
    # call the pca function to get reduced dimension, provided the num_components
    num_components = 3
    mat_reduced = PCA(X_transformed, num_components)

    # convert the given reduced matrix to a data frame
    principal_df = pd.DataFrame(mat_reduced , columns = ["PC" + str(i + 1) for i in range(num_components)])
    principal_df = pd.concat([principal_df , pd.DataFrame(y)] , axis = 1)
    fig=plt.figure(figsize=(6,6))
    
    # plot the graph below
    ax=fig.add_subplot(2,2,1)
    ax.set_title('PCA 1D Graph',fontsize=10)
    create1DGraph(ax, principal_df)

    ax2=fig.add_subplot(2,2,2)
    ax2.set_title('PCA 2D Graph',fontsize=10)
    create2DGraph(ax2, principal_df)

    ax3=fig.add_subplot(2,2,3, projection='3d')
    ax3.set_title('PCA 3D Graph',fontsize=10)
    create3DGraph(ax3, principal_df)
    fig.tight_layout()

def use_lda():
    df = pd.read_csv('Iris_data.csv')
    df = df.drop(['Id'], axis=1)

    # do data processing and mapping
    map_value = { 'Iris-setosa': 0, 'Iris-versicolor': 1, 'Iris-virginica': 2}
    df['Species'].replace(map_value,  inplace=True)
    X, y = df.drop('Species', axis=1), df['Species']

    # given num_classes it will generate the reduced dimension by calling and passing it to lda
    num_classes = 2
    projected_data = LDA(X, y, num_classes)
    projected_data1, projected_data2 = projected_data[:, 0], projected_data[:, 1]
    
    # plot the 1d graph
    fig=plt.figure(figsize=(6,6))
    ax=fig.add_subplot(2,2,1)
    ax.scatter(projected_data1, np.zeros(X.shape[0]), c=y, cmap="copper")
    ax.set_title('LDA 1D Graph',fontsize=10)
    plt.xlabel("LD 1")   

    # plot 2d graph
    ax2=fig.add_subplot(2,2,2)
    ax2.scatter(projected_data1, projected_data2, c=y, cmap="copper")
    plt.xlabel("LD 1")
    plt.ylabel("LD 2")
    ax2.set_title('LDA 2D Graph',fontsize=10)

    fig.tight_layout()
    plt.show()


use_pca()
use_lda()