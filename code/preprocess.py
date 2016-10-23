# preprocess.py
# @author: Lisa Wang
# @created: Oct 1 2016
#
#===============================================================================
# DESCRIPTION:
#
# This module creates the student id to trajectory id map and the trajectory id
# to asts map for a specified hoc number.
# Functions in preprocess should normally only be called once, e.g. to convert
# data into a different format. They are not part of the pipeline.
#
#===============================================================================
# CURRENT STATUS: Working
#===============================================================================
# USAGE:
# from preprocess import *
#
# Commandline:
# python preprocess.py -n <hoc_num>
# or to get help: python preprocess.py -h
#

import os, sys, getopt
import pickle
import csv
from collections import defaultdict

from path_names import *
from utils import *


def create_ast_id_to_program_embedding_map(hoc_num):
    # Use a map since ast ids are not necessarily consecutive. There are gaps. (e.g. hoc 18 doesn't have ast 15)
    program_embeddings_map = defaultdict()
    with open('../data/encoded_{}.csv'.format(hoc_num)) as f:
        reader = csv.reader(f)
        for i, row in enumerate(reader):
            assert(len(row) == 51), "row has unexpected length"
            ast_id = int(row[0])
            embedding = np.array([float(v) for v in row[1:]])
            program_embeddings_map[ast_id] = embedding
    return program_embeddings_map



def create_student_to_traj_map(hoc_num):
    student_to_traj_map = defaultdict()
    traj_dir_path = trajectories_dir_path(hoc_num)
    if os.path.exists(traj_dir_path):
        with open(os.path.join(traj_dir_path, 'idMap.txt')) as f:
            reader = csv.reader(f)
            headers = reader.next()
            for row in reader:
                student_id, traj_id = int(row[0]), int(row[1])
                student_to_traj_map[student_id] = traj_id
    return student_to_traj_map


def create_traj_to_asts_map(hoc_num):
    traj_to_asts_map = defaultdict(list)
    traj_dir_path = trajectories_dir_path(hoc_num)
    if os.path.exists(traj_dir_path):
        traj_files = [f for f in os.listdir(traj_dir_path) if (os.path.isfile(os.path.join(traj_dir_path, f)) and f not in ['counts.txt', 'idMap.txt'])]

        for f_path in traj_files:
            with open(os.path.join(traj_dir_path, f_path)) as f:
                asts = [int(l) for l in f]
                traj_id = int(os.path.splitext(f_path)[0])

                traj_to_asts_map[traj_id] = asts
    return traj_to_asts_map


def create_traj_to_total_steps_map(hoc_num):
    """
    For specified hoc, creates a map from trajectory id to the length of the trajectory.
    Parameters
    ----------
    hoc_num

    """
    traj_to_asts_map = get_traj_to_asts_map(hoc_num)
    traj_to_total_steps_map = {}
    for k, v in traj_to_asts_map.iteritems():
        traj_to_total_steps_map[k] = len(v)
    return traj_to_total_steps_map


def create_trajectory_scores_map(hoc_num):

    traj_to_asts_map = get_traj_to_asts_map(hoc_num)
    # student_to_traj_map = get_student_to_traj_map(hoc_num)
    # # get trajectory counts
    # traj_to_count_map = get_traj_counts(hoc_num)
    #
    # # create map from ast id to poisson rate parameter
    # ast_id_to_poisson_rate = Counter()
    #
    # for traj_id, asts in traj_to_asts_map.iteritems():
    # #         print ("Traj ID: {}, length: {}".format(traj_id, len(asts)))
    #     if len(asts) > 0 and asts[-1] == 0: # if last AST in traj was correct solution and cur ast is actually in trajectory
    #         for ast in asts:
    #             ast_id_to_poisson_rate[ast] += traj_to_count_map[traj_id] # scale count by # students who took this traj
    #
    # filename = ast_to_poisson_rate_path(hoc_num)
    # check_if_path_exists_or_create(filename)
    # pickle.dump(ast_id_to_poisson_rate, open(filename, 'wb'))

    ast_id_to_poisson_rate = get_ast_counts(hoc_num)

    # maps a trajectory to a pathscore as defined in Piech et al.
    # "Autonomously Generating Hints by Inferring Problem Solving Policies"
    traj_to_score = {}
    for traj_id, asts in traj_to_asts_map.iteritems():
        score = 0
        for ast in asts:
            score += 1 / (float(ast_id_to_poisson_rate[ast] + 1))
        traj_to_score[traj_id] = score

    filename = traj_to_score_path(hoc_num)
    check_if_path_exists_or_create(filename)
    pickle.dump(traj_to_score, open(filename, 'wb'))
    return traj_to_score

def create_traj_to_ast_embeddings_map(hoc_num):
    """
    For specified hoc, creates a map from from trajectory id to a list of embeddings.
    Parameters
    ----------
    hoc_num

    Returns
    -------

    """
    #TODO: Complete when I have data from Chris
    pass



def create_maps(hoc_num):
    student_to_traj_map = create_student_to_traj_map(hoc_num)
    filename = student_to_traj_path(hoc_num)
    check_if_path_exists_or_create(filename)
    pickle.dump(student_to_traj_map, open(filename, 'wb'))

    traj_to_asts_map = create_traj_to_asts_map(hoc_num)
    filename = traj_to_asts_path(hoc_num)
    check_if_path_exists_or_create(filename)
    pickle.dump(traj_to_asts_map,
                open(filename, 'wb'))


#
# def create_labels_num_steps_left_to_completion(hoc_num):
#     """
#     For specified HOC Problem, this method creates the labels y.
#     y is a list, containing lists specifying the number of steps left to completion
#     for each trajectory.
#
#     Parameters
#     ----------
#     hoc_num
#
#     Returns
#     -------
#
#     """
def usage():
    print ("python preprocess.py -n <hoc_num>")


def main():
    try:
        opts, args = getopt.getopt(sys.argv[1:],"hn:", ["help", "hoc_num="])
    except getopt.GetoptError as err:
        # print help information and exit:
        print str(err)  # will print something like "option -a not recognized"
        usage()
        sys.exit(2)
    for o, a in opts:
        if o in ("-h", "--help"):
            usage()
            sys.exit()
        elif o in ("-n", "--hoc_num"):
            hoc_num = int(a)
            create_maps(hoc_num)
        else:
            assert False, "unhandled option"

if __name__ == '__main__':
    for hoc_num in [18]:
        ast_id_to_program_embedding_map = create_ast_id_to_program_embedding_map(hoc_num)
        filename = ast_id_to_program_embedding_path(hoc_num)
        check_if_path_exists_or_create(filename)
        pickle.dump(ast_id_to_program_embedding_map,
                    open(filename, 'wb'))
    # main()





