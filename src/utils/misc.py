import re
from ntpath import basename
from os.path import join
from pathlib import Path

import numpy as np


def nums(string):
    return re.sub("[^0-9]", "", string)


def check_image(glob_list):
    if isinstance(glob_list, list):
        if len(glob_list) < 1:
            raise ValueError("No file found")
        elif len(glob_list) > 1:
            raise ValueError("More than one file found")


def not_all_exclude(cases, not_all, exclude):
    """Get the cases to run depending on the not_all and exclude parameters"""
    # If both not_all and exclude are not empty raise an error
    if not_all and exclude:
        raise ValueError("Both not_all and exclude cannot be used at the same time")
    # Just run for the cases in not_all
    if not_all:
        cases = [i for i in cases if basename(i).split(".")[0] in not_all]
    # Exclude the cases in exclude
    if exclude:
        cases = [i for i in cases if basename(i).split(".")[0] not in exclude]
    return cases


def join_create(*paths):
    """Returns joined path and creates the folder"""

    # Check if input is tuple of lists exclusively or string exclusively
    assert all(isinstance(i, list) for i in paths) or all(isinstance(i, str) for i in paths)

    # If input tuple of lists
    if all(isinstance(i, list) for i in paths):
        path = []
        for path in paths:
            ps = join(*path)
            Path(ps).mkdir(parents=True, exist_ok=True)
            path += [ps]
    # If input tuple of strings
    else:
        path = join(*paths)
        Path(path).mkdir(parents=True, exist_ok=True)

    return path


def Normalize(data):
    return (data - np.min(data)) / (np.max(data) - np.min(data))
