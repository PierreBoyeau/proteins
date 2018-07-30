from sklearn.model_selection import GroupShuffleSplit, ShuffleSplit


RANDOM_STATE = 42


def group_shuffle_indices(X, y, groups=None, **kwargs):
    return next(GroupShuffleSplit(random_state=RANDOM_STATE, test_size=0.2, **kwargs).split(X, y, groups))


def shuffle_indices(X, y, groups=None, **kwargs):
    return next(ShuffleSplit(random_state=RANDOM_STATE, test_size=0.2, **kwargs).split(X, y, groups))


