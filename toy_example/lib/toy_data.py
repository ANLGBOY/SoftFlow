import numpy as np
import sklearn
import sklearn.datasets
from sklearn.utils import shuffle as util_shuffle


# Dataset iterator
def inf_train_gen(data, rng=None, batch_size=200):
    if rng is None:
        rng = np.random.RandomState()

    if data == "2spirals_1d":
        n = np.sqrt(np.random.rand(batch_size // 2, 1)) * 540 * (2 * np.pi) / 360
        d1x = -np.cos(n) * n
        d1y = np.sin(n) * n
        x = np.vstack((np.hstack((d1x, d1y)), np.hstack((-d1x, -d1y)))) / 3
        return x

    elif data == "2spirals_2d":
        n = np.sqrt(np.random.rand(batch_size // 2, 1)) * 540 * (2 * np.pi) / 360
        d1x = -np.cos(n) * n + np.random.rand(batch_size // 2, 1) * 0.5
        d1y = np.sin(n) * n + np.random.rand(batch_size // 2, 1) * 0.5
        x = np.vstack((np.hstack((d1x, d1y)), np.hstack((-d1x, -d1y)))) / 3
        x += np.random.randn(*x.shape) * 0.1
        return x

    elif data == "swissroll_1d":
        data = sklearn.datasets.make_swiss_roll(n_samples=batch_size, noise=0.0)[0]
        data = data.astype("float32")[:, [0, 2]]
        data /= 5
        return data
    
    elif data == "swissroll_2d":
        data = sklearn.datasets.make_swiss_roll(n_samples=batch_size, noise=1.0)[0]
        data = data.astype("float32")[:, [0, 2]]
        data /= 5
        return data

    elif data == "circles_1d":
        data = sklearn.datasets.make_circles(n_samples=batch_size, factor=.5, noise=0.0)[0]
        data = data.astype("float32")
        data *= 3
        return data

    elif data == "circles_2d":
        data = sklearn.datasets.make_circles(n_samples=batch_size, factor=.5, noise=0.08)[0]
        data = data.astype("float32")
        data *= 3
        return data

    elif data =="2sines_1d":
        x = (rng.rand(batch_size) -0.5) * 2 * np.pi
        u = (rng.binomial(1,0.5,batch_size) - 0.5) * 2
        y = u * np.sin(x) * 2.5
        return np.stack((x, y), 1)

    elif data =="target_1d":
        shapes = np.random.randint(7, size=batch_size)
        mask = []
        for i in range(7):
            mask.append((shapes==i)*1.)

        theta = np.linspace(0, 2 * np.pi, batch_size, endpoint=False)
        x = (mask[0] + mask[1] + mask[2]) * (rng.rand(batch_size) -0.5) * 4 +\
         (-mask[3] + mask[4]*0.0 + mask[5]) * 2 * np.ones(batch_size) +\
         mask[6] * np.cos(theta)

        y = (mask[3] + mask[4] + mask[5]) * (rng.rand(batch_size) -0.5) * 4 +\
         (-mask[0] + mask[1]*0.0 + mask[2]) * 2 * np.ones(batch_size) +\
         mask[6] * np.sin(theta)

        return np.stack((x, y), 1)

    else:
        return inf_train_gen("2spirals_1d", rng, batch_size)
