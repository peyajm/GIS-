import numpy as np
from scipy.spatial import cKDTree
import pandas as pd
import matplotlib.pyplot as plt
import os
absolute_path = os.path.abspath(os.path.dirname('pollution_data.csv'))
df = pd.read_csv(absolute_path + '/pollution_data.csv')
df1 = df[['deviceid', 'pm25','x', 'y']]
class tree(object):
    def __init__(self, X=None, z=None, leafsize=10):
        if not X is None:
            self.tree = cKDTree(X, leafsize=leafsize )
        if not z is None:
            self.z = np.array(z)
    def fit(self, X=None, z=None, leafsize=10):
        return self.__init__(X, z, leafsize)
    def __call__(self, X, k=6, eps=1e-6, p=2, regularize_by=1e-9):
        self.distances, self.idx = self.tree.query(X, k, eps=eps, p=p)
        self.distances += regularize_by
        weights = self.z[self.idx.ravel()].reshape(self.idx.shape)
        mw = np.sum(weights/self.distances, axis=1) / np.sum(1./self.distances, axis=1)
        return mw
    def transform(self, X, k=23, p=2, eps=1e-6, regularize_by=1e-9):
        return self.__call__(X, k, eps, p, regularize_by)
data = df1.values
X1 = data[:,2:4]
z1 = data[:,1]
idw_tree = tree(X1, z1)
spacing1 = np.linspace(-1.32,0.6,200)
spacing2 = np.linspace(-0.48,0.8,100)
X2 = np.meshgrid(spacing1, spacing2)
grid_shape = X2[0].shape
X2 = np.reshape(X2, (2, -1)).T
z2 = idw_tree(X2)
fig, (ax2, ax3) = plt.subplots(1,2, sharex=True, sharey=True, figsize=(10,3))
ax2.scatter(X1[:,0], X1[:,1], c=z1, linewidths=0, cmap="YlOrBr")
ax2.set_title('Samples')
ax3.contourf(spacing1, spacing2, z2.reshape(grid_shape), cmap="YlOrBr")
im=ax3.contourf(spacing1, spacing2, z2.reshape(grid_shape), cmap="YlOrBr")
ax3.set_title('IDW interpolation')
plt.colorbar(im)
plt.show()
