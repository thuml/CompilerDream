import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from .. import core
from ..core import dists


class MLP(core.Module):

    def __init__(self, shape, layers, units, act="elu", norm="none", dropout=0.0, **out):
        super().__init__()
        self._shape = (shape,) if isinstance(shape, int) else shape
        self._layers = layers
        self._units = units
        self._norm = norm
        self._act = get_act(act)
        self._dropout = dropout
        self._out = out

    def __call__(self, features):
        x = features
        x = x.reshape([-1, x.shape[-1]])
        for index in range(self._layers):
            x = self.get(f"dense{index}", nn.Linear, x.shape[-1], self._units)(x)
            x = self.get(f"norm{index}", NormLayer, self._norm, x.shape[-1:])(x)
            x = self.get(f"dropout{index}", nn.Dropout1d, p=self._dropout)(x)
            x = self._act(x)
        x = x.reshape([*features.shape[:-1], x.shape[-1]])
        return self.get("out", DistLayer, self._shape, **self._out)(x)


class DistLayer(core.Module):

    def __init__(self, shape, dist="mse", min_std=0.1, init_std=0.0, unimix=0.0, bins=255, temp=1.0):
        super(DistLayer, self).__init__()
        self._shape = shape  # shape can be [], its equivalent to 1.0 in np.prod
        if dist == "twohot":
            self._shape = (3,)
        self._dist = dist
        self._min_std = min_std
        self._init_std = init_std

        # for dreamerv3
        self._bins = bins
        self._unimix = unimix
        self._temp = temp

    def __call__(self, inputs):
        shape = self._shape
        if self._dist.endswith('disc'):
            shape = (*self._shape, self._bins)
        out = self.get("out", nn.Linear, inputs.shape[-1], int(np.prod(shape)))(inputs)
        out = out.reshape([*inputs.shape[:-1], *shape])
        if self._dist in ("normal", "tanh_normal", "trunc_normal"):
            std = self.get("std", nn.Linear, inputs.shape[-1], int(np.prod(self._shape)))(inputs)
            std = std.reshape([*inputs.shape[:-1], *self._shape])
        if self._dist == "mse":
            dist = dists.MSE(out)
            return dists.Independent(dist, len(self._shape))
        if self._dist == "mae":
            dist = dists.MAE(out)
            return dists.Independent(dist, len(self._shape))
        if self._dist == 'disc':
            return dists.DiscDist(logits=out, low=-3, high=3, transbwd=lambda x: x, transfwd=lambda x: x)
        if self._dist == "normal":
            # NOTE Doesn't make sense std to be negative
            # NOT USED in algorithm
            raise NotImplementedError(self._dist)

            # dist = dists.Normal(out, std) # FIXME std can only be positive
            # return dists.Independent(dist, len(self._shape))
        if self._dist == "binary":
            # NOTE log_prob means binary_cross_entropy_with_logits
            dist = dists.Bernoulli(logits=out, validate_args=False)  # FIXME: validate_args=None? => Error
            return dists.Independent(dist, len(self._shape))
        if self._dist == "tanh_normal":
            # FIXME NOT USED in algorithm
            raise NotImplementedError(self._dist)

            # mean = 5 * torch.tanh(out / 5)
            # std = F.softplus(std + self._init_std) + self._min_std
            # dist = tdist.Normal(mean, std)
            # dist = tdist.TransformedDistribution(dist, common.TanhBijector()) #tfd.TransformedDistribution(dist, common.TanhBijector())
            # dist = tdist.Independent(dist, len(self._shape))
            # return common.SampleDist(dist)
        if self._dist == "trunc_normal":
            std = 2 * torch.sigmoid((std + self._init_std) / 2) + self._min_std
            dist = dists.TruncNormalDist(torch.tanh(out), std, -1, 1)
            return dists.Independent(dist, 1)
        if self._dist == "onehot":
            if self._unimix:
                probs = F.softmax(out/self._temp, -1)
                uniform = torch.ones_like(probs) / probs.shape[-1]
                probs = (1 - self._unimix) * probs + self._unimix * uniform
                out = torch.log(probs)
            dist = dists.OneHotDist(logits=out)
            return dist
        raise NotImplementedError(self._dist)


class NormLayer(core.Module):

    def __init__(self, name, normalized_shape):
        super().__init__()
        if name == "none":
            self._layer = None
        elif name == "layer":
            self._layer = nn.LayerNorm(normalized_shape, eps=1e-3)  # eps equal to tf
        else:
            raise NotImplementedError(name)

    def __call__(self, features):
        if not self._layer:
            return features
        return self._layer(features)


def get_act(act):
    if isinstance(act, str):
        name = act
        if name == "none":
            return lambda x: x
        if name == "mish":
            return lambda x: x * torch.tanh(F.softplus(x))
        elif hasattr(F, name):
            return getattr(F, name)
        elif hasattr(torch, name):
            return getattr(torch, name)
        else:
            raise NotImplementedError(name)
    else:
        return act
