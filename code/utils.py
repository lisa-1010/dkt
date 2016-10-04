# utils.py
# @author: Lisa Wang
# @created: Oct 2 2016
#
#===============================================================================
# DESCRIPTION:
#
#===============================================================================
# CURRENT STATUS: Working
#===============================================================================
# USAGE:
# from utils import *


import os, sys, getopt
import pickle
import csv
from collections import defaultdict
import numpy as np

from path_names import *


def get_student_to_traj_map(hoc_num):
    student_to_traj_map = pickle.load(open(student_to_traj_path(hoc_num)))
    return student_to_traj_map


def get_traj_to_asts_map(hoc_num):
    traj_to_asts_map = pickle.load(open(traj_to_asts_path(hoc_num)))
    return traj_to_asts_map


def get_traj_to_total_steps_map(hoc_num):
    traj_to_total_steps_map = pickle.load(open(traj_to_total_steps_path(hoc_num)))
    return traj_to_total_steps_map


if __name__ == "__main__":
    m = get_traj_to_total_steps_map(18)
    print m.items()[:100]



