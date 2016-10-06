# utils.py
# @author: Lisa Wang
# @created: Oct 2 2016
#
#===============================================================================
# DESCRIPTION:
#
#===============================================================================
# CURRENT STATUS: In progress
#===============================================================================
# USAGE:
# from utils import *


import os, sys, getopt
import pickle
import csv
from collections import defaultdict
import numpy as np
from tflearn.data_utils import to_categorical, pad_sequences
import random
from path_names import *


def check_if_path_exists_or_create(file_path):
    if not os.path.exists(os.path.dirname(file_path)):
        try:
            os.makedirs(os.path.dirname(file_path))
        except OSError as exc:  # Guard against race condition
            return False
    return True


def load_x_char_level_ast_vecs(hoc_num):
    pass


def get_program_embedding(hoc_num, ast_id):

    # TODO: to be completed when I get ast -> encoding mappings from Chris
    # Replace random vector below
    return np.random.rand(3)


def load_x_program_embeddings(hoc_num, traj_ids=None, max_timesteps=10):
    # if traj_ids are provided, only compute embeddings for specified trajectories.
    #  Otherwise, compute embeddings for all trajectories.
    # x will be a list of lists.
    # Each row in x is a list of embedding vectors. The length of each list can vary.

    traj_to_asts_map = get_traj_to_asts_map(hoc_num)
    x = []
    if traj_ids == None:
        num_traj = len(traj_to_asts_map.keys())
        for traj_id in xrange(num_traj):
            if traj_id in traj_to_asts_map:
                ast_ids = traj_to_asts_map[traj_id] # list of ast ids in the trajectory
                traj_len = len(ast_ids)
                # embeddings = [get_program_embedding(hoc_num, ast_id) for i, ast_id in enumerate(ast_ids) if i < max_timesteps]
                embeddings = []

                for t in xrange(max_timesteps):
                    if t < traj_len:
                        embeddings.append(get_program_embedding(hoc_num, ast_ids[t]))
                x.append(embeddings)
            else:
                print ("trajectory with id {} not found. ".format(traj_id))
    print ("Printing first 10 entries of x. Each entry should be a list of embedding vectors. \
    the length of each list can vary, but should not exceed {}.".format(max_timesteps))
    return x


# def load_data_steps_to_completion(hoc_num):
#     # Only applicable to trajectories that actually finish with the correct solution
#     # returns (x,y)
#     # x: np matrix of shape (num_samples, num_timesteps, embedding_dim)
#     # y: np matrix of shape (num_samples, num_timesteps, max_steps_to_completion)
#     traj_to_total_steps_map = get_traj_to_total_steps_map(hoc_num)
#     pass
#
#
# def load_data_steps_to_giving_up(hoc_num):
#     # Only applicable to trajectories that don't end with the right solution
#     # similar to load_data_steps_to_completion
#     pass

#===============================================================================
# Below: data where y is not a time series.
#===============================================================================


def load_data_will_student_solve_next_problem(hoc_num):
    """

    Parameters
    ----------
    hoc_num

    Returns
    -------
    x: np matrix of shape (num_samples, num_timesteps, embedding_dim)
    y: np matrix of shape (num_samples, 2) (binary one-hot encoding)
    """

    # x = load_x_embeddings_for_all_trajectories(hoc_num)

    student_to_traj_map = get_student_to_traj_map(hoc_num)
    all_students = student_to_traj_map.keys()



# ===============================================================================
# Helper functions to load maps
#===============================================================================

def get_students_who_solved_next_problem(hoc_num):
    # returns list of all students who successfully solved the next problem after the specified hoc.
    students_who_solved_next_problem = []
    with open('../data/hoc{}/nextProblem/perfectSet.txt'.format(hoc_num)) as f:
        for line in f:
            student_id = int(line)
            students_who_solved_next_problem.append(student_id)
    return students_who_solved_next_problem


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
    # m = get_traj_to_total_steps_map(18)
    # print m.items()[:100]
    # students = get_students_who_solved_next_problem(18)
    # print students[:10]

    x = load_x_program_embeddings(18)
    for i in xrange(100,110):
        print (len(x[i]))
        print (x[i])



