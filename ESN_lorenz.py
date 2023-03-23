import numpy as np
import reservoirpy
from  matplotlib import pyplot as plt
from reservoirpy.datasets import lorenz
from reservoirpy.nodes import Reservoir, Ridge, Input

#For reproducability:
reservoirpy.set_seed(42)


fig = plt.figure()
trainSize = 95000

X = np.load("xyz_coordinates.npy")
X = np.transpose(X)
t = np.arange(len(X)-trainSize)

plt.title("Test data")
plt.plot(t,X[trainSize:,1])
plt.ylabel("y(t)")
plt.xlabel("t")

time2 = np.arange(4999)

reservoir1 = Reservoir(1000, lr=0.3, sr=1.1)
reservoir2 = Reservoir(1000, lr=0.3, sr=1.1)
readout = Ridge(ridge=1e-6)
esn = reservoir1 >> reservoir2>> readout
esn.fit(X[:trainSize-1],X[1:trainSize],warmup=100)
predictions = esn.run(X[trainSize:-1])

plt.plot(time2,predictions[:,1],r'-', linewidth=1,)
plt.ylim((-25,25))

fig2 = plt.figure()

difference = np.abs(predictions[:,1]-X[95001:,1])


plt.plot(time2,difference,linewidth=0.5)
plt.title("test vs prediction difference")
