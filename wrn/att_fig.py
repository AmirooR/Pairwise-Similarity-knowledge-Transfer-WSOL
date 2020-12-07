import numpy as np
import matplotlib.pyplot as plt

def get_points(mean, std=.15, npoints=10):
  x = np.random.normal(mean[0], std, npoints)
  y = np.random.normal(mean[1], std, npoints)
  return np.stack([x,y]).T

def att_mean(points, ref, sigma=.15):
  ref = np.array(ref)[None]
  dists = points - ref
  dists = np.sum(dists**2, axis=1)
  w = np.exp(-dists/sigma/2)
  w = w/np.sum(w)
  att_mean = points * w[:, None]
  return att_mean.sum(axis=0)


p0 = [-1, .5]
p1 = [.5, 0]
p2 = [0, .5]

a = [0, 1]
b = [1, 0]
c = [-1, 0]
points_cluster = [get_points(x) for x in [a,b,c]]
points = np.concatenate(points_cluster, axis=0)


mean = np.mean(points, axis=0)
att0 = att_mean(points, p0)
att1 = att_mean(points, p1)
att2 = att_mean(points, p2)

plt.plot(points_cluster[0][:, 0], points_cluster[0][:, 1], '.', markersize=20, color='purple')
plt.plot(points_cluster[1][:, 0], points_cluster[1][:, 1], '.', markersize=20, color='green')
plt.plot(points_cluster[2][:, 0], points_cluster[2][:, 1], '.', markersize=20, color='pink')

markers = ['h', 's', 's', 's', 'o', 'o', 'o']
colors = ['black', 'red', 'blue', 'green', 'red', 'blue', 'green']
for i, p in enumerate([mean, att0, att1, att2, p0, p1, p2]):
  plt.plot(p[0], p[1], markers[i], color=colors[i], markersize=20)

plt.gca().axes.get_yaxis().set_visible(False)
plt.gca().axes.get_xaxis().set_visible(False)
plt.show()
#plt.xlim(0, 1.8)
