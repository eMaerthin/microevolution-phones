# test tsne
import numpy as np
from matplotlib import pyplot as plt
from dimensionality_reduction import fit_tsne
from pydiffmap import diffusion_map as dm

#mydmap = dm.DiffusionMap.from_sklearn(n_evecs = 2, alpha = 0.9, epsilon = 'bgh', k=64)

#neighbor_params = {'n_jobs': -1, 'algorithm': 'ball_tree'}
#mydmap = dm.DiffusionMap.from_sklearn(n_evecs=2, k=10, epsilon='bgh', alpha=1.0, neighbor_params=neighbor_params)

Nc = 200
var = 1
dims = 100
zers = np.zeros(dims)
y0 = zers.copy()
y0[0] = 20
y0[1] = 0
z0 = zers.copy()
z0[0] = 40
z10 = zers.copy()
z10[0] = 40
z10[1] = 0
x = np.random.multivariate_normal(np.zeros(dims), np.diag(np.ones(dims)*0.1*var), Nc)
y = np.random.multivariate_normal(y0, np.diag(np.ones(dims)*var), Nc)
z = np.random.multivariate_normal(z0, np.diag(np.ones(dims)*var), Nc)
z1 = np.random.multivariate_normal(z10, np.diag(np.ones(dims)*var), Nc)
#w = np.linspace(np.array([0, 0, 0, 0]), np.array([100, 0, 0, 0]), 1000, axis=0)
#l1 = np.linspace(-3, 38, 6)
#l2 = np.linspace(-10, 10, 20)
#l3 = np.linspace(-10, 10, 20)
# l4 = np.linspace(-10, 10, 20)
#l0 = np.linspace(-3, 3, 6)
#mesh = np.meshgrid(l1, l0, l0, l0, l0, l0, l0, l0, l0, l0)
#arr = np.array(mesh)
#arr_moved = np.moveaxis(arr, 0, -1)
#w = np.reshape(arr_moved, (243 * 243 * 1024, 10))
v_mult = np.ones(dims) * 10
v_mult[0] = 50
#v_mult[1] = 25
v_subt = np.ones(dims) * 5
#v_subt[0] = 30
w = np.random.sample((2000, dims)) * v_mult - v_subt
print(w.shape)
#idx = np.random.choice(w.shape[0], 500, replace=False)
#w = w[idx, :]
#print(w)
#print(w.shape)
# w = np.random.multivariate_normal(np.array([140, 0, 0, 0]), np.diag(np.ones(4)*var), Nc)
t = np.concatenate((x, y, z, z1, w))
t0 = np.concatenate((x, y, z, z1))
#dmap = mydmap.fit_transform(t)
# alternatively:
# x=np.random.multivariate_normal(np.zeros(4), np.diag(np.ones(4)*var), Nc)
# t=x.copy()
# for i in range(10):
#     r=np.random.multivariate_normal(np.array([10+10*i,0,0,0]), np.diag(np.ones(4)*var), Nc)
#     t=np.concatenate((t, r))
tsne_iters = 1000
tsne_perplexity = 50
out = fit_tsne(t, n_iter=tsne_iters, perplexity=tsne_perplexity)
out0 = fit_tsne(t0, n_iter=tsne_iters, perplexity=tsne_perplexity)

fig, axs = plt.subplots(2, 2, constrained_layout=True)
print(axs.shape)
fig.suptitle(f'Input data: {dims}D.\n blue - \"supporting\" random sample uniformly distributed ({w.shape[0]} points)\n other colors - 4 multivariate normal clusters x {Nc} points each\n Clusters are zero-centered except for first two dims, with variance: {var}')
axs[0][0].set_title('input data\n(Visualized just first 2 dims)')
axs[0][0].plot(w[..., 0], w[..., 1], 'ob')
axs[0][0].plot(x[..., 0], x[..., 1], 'ok')
axs[0][0].plot(y[..., 0], y[..., 1], 'or')
axs[0][0].plot(z[..., 0], z[..., 1], 'oy')
axs[0][0].plot(z1[..., 0], z1[..., 1], 'og')
#plt.subplot(222)
#plt.title('tsne')
axs[0][1].set_title(f'tsne image (2D),\n iters: {tsne_iters}, perplexity: {tsne_perplexity}')
axs[0][1].plot(out[4*Nc:, 0], out[4*Nc:, 1], 'ob')
axs[0][1].plot(out[:Nc, 0], out[:Nc, 1], 'ok')
axs[0][1].plot(out[Nc:2*Nc, 0], out[Nc:2*Nc, 1], 'or')
axs[0][1].plot(out[2*Nc:3*Nc, 0], out[2*Nc:3*Nc, 1], 'oy')
axs[0][1].plot(out[3*Nc:4*Nc, 0], out[3*Nc:4*Nc, 1], 'og')

axs[1][0].set_title('input data without \"support\"\n(Visualized just first 2 dims)')
axs[1][0].plot(x[..., 0], x[..., 1], 'ok')
axs[1][0].plot(y[..., 0], y[..., 1], 'or')
axs[1][0].plot(z[..., 0], z[..., 1], 'oy')
axs[1][0].plot(z1[..., 0], z1[..., 1], 'og')
#plt.subplot(222)
#plt.title('tsne')
axs[1][1].set_title(f'tsne image without \"support\", 2D,\n iters: {tsne_iters}, perplexity: {tsne_perplexity}')
axs[1][1].plot(out0[:Nc, 0], out0[:Nc, 1], 'ok')
axs[1][1].plot(out0[Nc:2*Nc, 0], out0[Nc:2*Nc, 1], 'or')
axs[1][1].plot(out0[2*Nc:3*Nc, 0], out0[2*Nc:3*Nc, 1], 'oy')
axs[1][1].plot(out0[3*Nc:4*Nc, 0], out0[3*Nc:4*Nc, 1], 'og')
#
# plt.subplot(313)
# plt.title('diffusion map')
# plt.plot(dmap[4*Nc:, 0], dmap[4*Nc:, 1], 'ob')
# plt.plot(dmap[3*Nc:4*Nc, 0], dmap[3*Nc:4*Nc, 1], 'Xg')
# plt.plot(dmap[2*Nc:3*Nc, 0], dmap[2*Nc:3*Nc, 1], 'Xy')
# plt.plot(dmap[Nc:2*Nc, 0], dmap[Nc:2*Nc, 1], 'Pr')
# plt.plot(dmap[:Nc, 0], dmap[:Nc, 1], 'Xk')

plt.show()