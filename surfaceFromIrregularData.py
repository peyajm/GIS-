# Copyright (c) 2009, Enthought, Inc.
# License: BSD Style.

import numpy as np
import pandas as pd
import os
# Create data with x and y random in the [-2, 2] segment, and z a
# Gaussian function of x and y.
np.random.seed(12345)
x = 4 * (np.random.random(500) - 0.5)
y = 4 * (np.random.random(500) - 0.5)
absolute_path = os.path.abspath(os.path.dirname('pollution_data.csv'))
df = pd.read_csv(absolute_path + '/pollution_data.csv')
data=df.values
X1=data[:,15]
Y1=data[:,16]
X1=np.array(X1, dtype='float64')
Y1=np.array(Y1, dtype='float64')
#X1=np.reshape(X1,-1)
Z1=data[:,7]
Z1=np.array(Z1)
def f(x, y):
    return np.exp(-(x ** 2 + y ** 2))
Z1=f(X1,Y1)
z = f(x, y)

print(x.dtype, X1.dtype)
print(x.shape, y.shape)
print(X1.shape, Y1.shape)
print(z.shape, Z1.shape) 

from mayavi import mlab
mlab.figure(1, fgcolor=(0, 0, 0), bgcolor=(1, 1, 1))

# Visualize the points
pts = mlab.points3d(X1,Y1,Z1,Z1, scale_mode='none', scale_factor=0.2)

# Create and visualize the mesh
mesh = mlab.pipeline.delaunay2d(pts)
surf = mlab.pipeline.surface(mesh)
pts.remove()
mlab.view(47, 57, 8.2, (0.1, 0.15, 0.14))
mlab.show()