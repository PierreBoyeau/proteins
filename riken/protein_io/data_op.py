from sklearn.model_selection import GroupShuffleSplit, ShuffleSplit, LeaveOneGroupOut
import pandas as pd

RANDOM_STATE = 42

"""
    Contains functions used to do datasets splits, like train/test or cross validation.
    group-specific functions can take into accounts appartenance to groups, like species in order
    to avoid to have very similar proteins in both train and test splits.
"""


def group_shuffle_indices(X, y, groups=None, test_size=0.2, **kwargs):
    """
    Train/Test split by taking groups into account.
    :param X:
    :param y:
    :param groups:
    :param test_size:
    :param kwargs:
    :return:
    """
    return next(GroupShuffleSplit(random_state=RANDOM_STATE, test_size=test_size, **kwargs)
                .split(X, y, groups))


def shuffle_indices(X, y, groups=None, **kwargs):
    """
    Usual Train/Test shuffle split.
    :param X:
    :param y:
    :param groups:
    :param kwargs:
    :return:
    """
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


