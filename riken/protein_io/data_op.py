from sklearn.model_selection import GroupShuffleSplit, ShuffleSplit, LeaveOneGroupOut
import pandas as pd

RANDOM_STATE = 42


def group_shuffle_indices(X, y, groups=None, test_size=0.2, **kwargs):
    return next(GroupShuffleSplit(random_state=RANDOM_STATE, test_size=test_size, **kwargs)
                .split(X, y, groups))


def shuffle_indices(X, y, groups=None, **kwargs):
    return next(ShuffleSplit(random_state=RANDOM_STATE, test_size=0.2, **kwargs)
                .split(X, y, groups))


def pseudo_cv_groups(X, y, groups=None, nb_per_group_thresh=79):
    """
    Creates pseudo-cv splits based on groups information:
        We apply KeepOneOut policy and we merge small groups together in a bigger group
    :param X:
    :param y:
    :param groups:
    :param nb_per_group_thresh:
    :return:
    """
    groups_df = pd.Series(groups).astype('category').cat.codes
    group_sizes = groups_df.groupby(groups_df).size()

    new_groups = groups_df.values.copy()
    # We merge groups which do not occur more than nb_per_group_thresh in a same group
    new_groups[group_sizes[groups_df].values <= nb_per_group_thresh] = -1
    new_groups = pd.Series(new_groups).astype('category').cat.codes

    print('Performing Cross-Validation with {} groups'.format(len(new_groups.unique())))

    return LeaveOneGroupOut().split(X, y, new_groups)


