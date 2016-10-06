# path_names.py
# @author: Lisa Wang
# @created: Oct 1 2016
#
#===============================================================================
# DESCRIPTION:
#
# Exports path names for files and directories, so paths are consistent across
# modules.
#
#===============================================================================
# CURRENT STATUS: Working
#===============================================================================
# USAGE: from path_names import *

def trajectories_dir_path(hoc_num):
    return '../data/hoc{}/trajectories/'.format(hoc_num)

def next_problem_dir_path(hoc_num):
    return '../data/hoc{}/nextProblem/'.format(hoc_num)

def student_to_traj_path(hoc_num):
    return '../preprocessed_data/hoc{}/student_id_to_trajectory_id_map.pickle'.format(hoc_num)

def traj_to_asts_path(hoc_num):
    return '../preprocessed_data/hoc{}/trajectory_id_to_asts_map.pickle'.format(hoc_num)


def traj_to_total_steps_path(hoc_num):
    return '../preprocessed_data/hoc{}/trajectory_id_to_total_steps_map.pickle'.format(hoc_num)
