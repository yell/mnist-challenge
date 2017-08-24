import numpy as np

import env
from utils import RNG


def get_initialization(initialization_name):
    for k, v in globals().items():
        if k.lower() == initialization_name.lower():
            return v
    raise ValueError("invalid initialization name '{0}'".format(initialization_name))


def _glorot_fan(shape):
    """
    Examples
    --------
    >>> shape = (2, 3, 4, 5)
    >>> _glorot_fan(shape)
    (60, 40)
    """
    assert len(shape) >= 2
    receptive_field_size = np.prod(shape[2:])
    fan_in  = receptive_field_size * shape[1]
    fan_out = receptive_field_size * shape[0]
    return fan_in, fan_out

def glorot_uniform(shape, random_seed=None):
    fan_in, fan_out = _glorot_fan(shape)
    s = np.sqrt(6. / (fan_in + fan_out))
    return RNG(random_seed).uniform(low=-s, high=s, size=shape)

def glorot_normal(shape, random_seed=None):
    fan_in, fan_out = _glorot_fan(shape)
    s = np.sqrt(2. / (fan_in + fan_out))
    return RNG(random_seed).normal(scale=s, size=shape)


if __name__ == '__main__':
    # run corresponding tests
    import env
    from utils.testing import run_tests
    run_tests(__file__)