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


from collections import defaultdict
import os, sys, getopt
import pickle
import csv, json
from pprint import pprint
from collections import defaultdict
import numpy as np
from tflearn.data_utils import to_categorical, pad_sequences
import random
from path_names import *


from collections import Counter

def check_if_path_exists_or_create(file_path):
    if not os.path.exists(os.path.dirname(file_path)):
        try:
            os.makedirs(os.path.dirname(file_path))
        except OSError as exc:  # Guard against race condition
            return False
    return True


def load_x_char_level_ast_vecs(hoc_num):
    pass


def missing_ast_id_embedding(dim, random_seed=47):
    np.random.seed(random_seed)
    return np.random.rand(dim)

def load_program_embeddings(hoc_num, max_timesteps=10):

    ast_id_to_program_embedding_map = get_ast_id_to_program_embedding_map(hoc_num)

    traj_to_asts_map = get_traj_to_asts_map(hoc_num)
    traj_ids = sorted(traj_to_asts_map.keys())

    embedding_dim = len(ast_id_to_program_embedding_map.values()[0])
    traj_id_to_embeddings_map = {}

    for traj_id in traj_ids:
        ast_ids = traj_to_asts_map[traj_id] # list of ast ids in the trajectory
        traj_len = len(ast_ids)

        embeddings = []
        for t in xrange(max_timesteps):
            if t < traj_len:
                ast_id = ast_ids[t]
                if ast_id in ast_id_to_program_embedding_map:
                    embeddings.append(ast_id_to_program_embedding_map[ast_id])
                else:
                    embeddings.append(missing_ast_id_embedding(dim=embedding_dim))
            else:
                embeddings.append(np.zeros(embedding_dim))

        traj_id_to_embeddings_map[traj_id] = embeddings
    return traj_id_to_embeddings_map



def get_all_student_ids(hoc_num):
    student_to_traj_map = get_student_to_traj_map(hoc_num)
    all_students = student_to_traj_map.keys()
    return all_students


# def load_data_steps_to_completion(hoc_num):
#     # Only applicable to trajectories that actually finish with the correct solution
#     # returns (x,y)
#     # x: np matrix of shape (num_samples, num_timesteps, embedding_dim)
#     # y: np matrix of shape (num_samples, num_timesteps, max_steps_to_completion)
#     traj_to_total_steps_map = get_traj_to_total_steps_map(hoc_num)
#     pass
#

#===============================================================================
# Below: data where y is not a time series.
#===============================================================================



def load_data_will_student_solve_next_problem(hoc_num, minlen=3, maxlen=10, y_is_seq=False, with_mask=False, return_student_ids=False):
    """
    Returns:
    x: list of embedding sequences. Each entry in the list is one sample corresponding to a student.
    y: list of binary truth values, indicating whether student solved next problem perfectly.
    """
    print ("Loading data...")
    traj_id_to_embeddings_map = load_program_embeddings(hoc_num, max_timesteps=maxlen)
    traj_to_asts_map = get_traj_to_asts_map(hoc_num)
    student_to_traj_map = get_student_to_traj_map(hoc_num)
    all_student_ids = sorted(student_to_traj_map.keys())

    students_who_solved_next_problem = set(get_students_who_solved_next_problem(hoc_num))

    x, y, mask, student_ids = [], [], [], []

    for student_id in all_student_ids:
        traj_id = int(student_to_traj_map[student_id])
        traj_len = len(traj_to_asts_map[traj_id])

        if traj_len >= minlen:
            x.append(traj_id_to_embeddings_map[traj_id])
            if student_id in students_who_solved_next_problem:
                y.append(1)
            else:
                y.append(0)
            if with_mask:
                # mask should traj_len 1's and rest to maxlen
                # e.g. if traj_len is 3, then cur_mask should be [1,1,1,0,0,0,0,0,0,0]
                cur_mask = np.lib.pad(np.ones(min(traj_len, maxlen)), (0, max(0, maxlen - traj_len)), 'constant',
                                      constant_values=(0))
                mask.append(cur_mask)
            if return_student_ids:
                student_ids.append(student_id)

    x = np.array(x)
    y = np.array(to_categorical(y, nb_classes=2))
    mask = np.array(mask)
    student_ids = np.array(student_ids)
    # mask = np.reshape(mask, (mask.shape[0], mask.shape[1], 1))

    if y_is_seq:
        y_seq = np.repeat(y, maxlen, axis=0)
        y = y_seq.reshape(y.shape[0], maxlen, y.shape[1])

    print ("Data loaded.")

    if with_mask:
        if return_student_ids:
            return x, y, mask, student_ids
        else:
            return x, y, mask

    if return_student_ids:
        return x, y, student_ids
    return x, y


def load_data_will_student_solve_next_problem_traj_len(hoc_num, only_traj_len=5, y_is_seq=False):
    """
    Returns:
    x: list of embedding sequences. Each entry in the list is one sample corresponding to a student.
    y: list of binary truth values, indicating whether student solved next problem perfectly.
    student_ids: corresponding student_ids for x and y. x[i] and y[i] correspond to student with student_ids[i]
    """
    print ("Loading data...")
    traj_id_to_embeddings_map = load_program_embeddings(hoc_num, max_timesteps=only_traj_len)
    traj_to_asts_map = get_traj_to_asts_map(hoc_num)
    student_to_traj_map = get_student_to_traj_map(hoc_num)
    all_student_ids = sorted(student_to_traj_map.keys())

    students_who_solved_next_problem = set(get_students_who_solved_next_problem(hoc_num))

    x, y, student_ids = [], [], []

    for student_id in all_student_ids:
        traj_id = int(student_to_traj_map[student_id])
        traj_len = len(traj_to_asts_map[traj_id])

        if traj_len == only_traj_len:
            x.append(traj_id_to_embeddings_map[traj_id])
            if student_id in students_who_solved_next_problem:
                y.append(1)
            else:
                y.append(0)
            student_ids.append(student_id)

    x = np.array(x)
    y = np.array(to_categorical(y, nb_classes=2))
    student_ids = np.array(student_ids)

    if y_is_seq:
        y_seq = np.repeat(y, only_traj_len, axis=0)
        y = y_seq.reshape(y.shape[0], only_traj_len, y.shape[1])

    print ("Data loaded.")
    return x, y, student_ids


def load_data_will_student_solve_next_problem_baseline(hoc_num, minlen=3, maxlen=10):
    """
    Returns:
    x: list of embedding sequences. Each entry in the list is one sample corresponding to a student.
    y: list of binary truth values, indicating whether student solved next problem perfectly.
    """
    # print ("Loading data...")
    traj_to_asts_map = get_traj_to_asts_map(hoc_num)
    traj_to_score_map = get_traj_to_score_map(hoc_num)
    student_to_traj_map = get_student_to_traj_map(hoc_num)

    all_student_ids = sorted(student_to_traj_map.keys())

    students_who_solved_next_problem = set(get_students_who_solved_next_problem(hoc_num))

    x, y = [], []

    for student_id in all_student_ids:
        traj_id = int(student_to_traj_map[student_id])
        traj_len = len(traj_to_asts_map[traj_id])

        if traj_len >= minlen:
            x.append(traj_to_score_map[traj_id])
            if student_id in students_who_solved_next_problem:
                y.append(1)
            else:
                y.append(0)

    x, y  = np.array(x), np.array(y)
    x = x.reshape(-1,1) # since sklearn doesn't accept 1-d input.
    # print ("Data loaded.")
    return x, y


def load_data_will_student_solve_next_problem_baseline_pathscore_success_on_cur_problem(hoc_num, minlen=3, maxlen=10):
    """
    Returns:
    x: np.array  is one sample corresponding to a student.
    y: list of binary truth values, indicating whether student solved next problem perfectly.
    """
    # print ("Loading data...")
    traj_to_asts_map = get_traj_to_asts_map(hoc_num)
    traj_to_score_map = get_traj_to_score_map(hoc_num)
    student_to_traj_map = get_student_to_traj_map(hoc_num)

    all_student_ids = sorted(student_to_traj_map.keys())

    students_who_solved_next_problem = set(get_students_who_solved_next_problem(hoc_num))

    x, y = [], []

    for student_id in all_student_ids:
        traj_id = int(student_to_traj_map[student_id])
        traj_len = len(traj_to_asts_map[traj_id])

        if traj_len >= minlen:
            score = traj_to_score_map[traj_id]
            if traj_to_asts_map[traj_id][-1] == 0:  # if student's trajectory ends in correct solution
                success_on_cur_problem = 1
            else:
                success_on_cur_problem = 0
            x.append(np.array([score,success_on_cur_problem]))
            if student_id in students_who_solved_next_problem:
                y.append(1)
            else:
                y.append(0)

    x, y  = np.array(x), np.array(y)
    # print ("Data loaded.")
    return x, y


def load_data_will_student_solve_next_problem_baseline_pathscore_success_on_cur_problem_traj_len(hoc_num, only_traj_len=5):
    """
    Only includes students whose trajectory has a certain length, as specified by only_traj_len parameter.
    Returns:
    x: np.array of shape (n_students, 2) is one sample corresponding to a student. Two features, one is the pathscore,
    the other is indicator whether student solved current problem  perfectly.
    y: np.array of shape (n_students,) of binary truth values, indicating whether student solved next problem perfectly.
    """
    # print ("Loading data...")
    traj_to_asts_map = get_traj_to_asts_map(hoc_num)
    traj_to_score_map = get_traj_to_score_map(hoc_num)
    student_to_traj_map = get_student_to_traj_map(hoc_num)

    all_student_ids = sorted(student_to_traj_map.keys())

    students_who_solved_next_problem = set(get_students_who_solved_next_problem(hoc_num))

    x, y = [], []

    for student_id in all_student_ids:
        traj_id = int(student_to_traj_map[student_id])
        traj_len = len(traj_to_asts_map[traj_id])

        if traj_len == only_traj_len:
            score = traj_to_score_map[traj_id]
            if traj_to_asts_map[traj_id][-1] == 0:  # if student's trajectory ends in correct solution
                success_on_cur_problem = 1
            else:
                success_on_cur_problem = 0
            x.append(np.array([score,success_on_cur_problem]))
            if student_id in students_who_solved_next_problem:
                y.append(1)
            else:
                y.append(0)

    x, y  = np.array(x), np.array(y)
    # print ("Data loaded.")
    return x, y




def load_data_will_student_solve_current_problem(hoc_num, minlen=3, maxlen=10, y_is_seq=False):
    """
    Returns:
    x: list of embedding sequences. Each entry in the list is one sample corresponding to a student.
    y: list of binary truth values, indicating whether student solved current problem perfectly.
    """
    print ("Loading data...")
    traj_id_to_embeddings_map = load_program_embeddings(hoc_num, max_timesteps=maxlen)
    student_to_traj_map = get_student_to_traj_map(hoc_num)
    traj_to_asts_map = get_traj_to_asts_map(hoc_num)
    all_student_ids = sorted(student_to_traj_map.keys())
    x, y = [], []

    for student_id in all_student_ids:
        traj_id = int(student_to_traj_map[student_id])
        traj_len = len(traj_to_asts_map[traj_id])
        if traj_len >= minlen:
            x.append(traj_id_to_embeddings_map[traj_id])
            if traj_to_asts_map[traj_id][-1] == 0: # if student's trajectory ends in correct solution
                y.append(1)
            else:
                y.append(0)

    x = np.array(x)
    y = np.array(to_categorical(y, nb_classes=2))

    if y_is_seq:
        y_seq = np.repeat(y, maxlen, axis=0)
        y = y_seq.reshape(y.shape[0], maxlen, y.shape[1])

    print ("Data loaded.")

    return x, y


def print_all_asts_in_traj(hoc_num, traj_id, filename=None):
    traj_to_asts_map = get_traj_to_asts_map(hoc_num)
    asts = traj_to_asts_map[traj_id]
    traj_json = []
    if filename:
        f = open(filename, 'wb+')
    for ast_id in asts:
        ast_content = read_ast(hoc_num, ast_id)
        print ast_id
        traj_json.append({ast_id: ast_content})
        pprint(ast_content)

    if filename:
        f.write(json.dumps({traj_id: traj_json}))
        f.close()



def read_ast(hoc_num, ast_id):
    ast_file_path = ('../data/hoc{}/asts/{}.json'.format(hoc_num, ast_id))
    print ast_file_path

    if os.path.isfile(ast_file_path):
        with open(ast_file_path, 'rb+') as f:
            ast_content = json.load(f)
        return ast_content
    else:
        return "AST {} was not found. ".format(ast_id)



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


def get_ast_id_to_program_embedding_map(hoc_num):
    # Use a map since ast ids are not necessarily consecutive. There are gaps. (e.g. hoc 18 doesn't have ast 15)
    program_embeddings_map = pickle.load(open(ast_id_to_program_embedding_path(hoc_num)))
    return program_embeddings_map


def get_student_to_traj_map(hoc_num):
    student_to_traj_map = pickle.load(open(student_to_traj_path(hoc_num)))
    return student_to_traj_map


def get_traj_to_asts_map(hoc_num):
    traj_to_asts_map = pickle.load(open(traj_to_asts_path(hoc_num)))
    return traj_to_asts_map


def get_traj_to_total_steps_map(hoc_num):
    traj_to_total_steps_map = pickle.load(open(traj_to_total_steps_path(hoc_num)))
    return traj_to_total_steps_map


def get_traj_to_score_map(hoc_num):
    traj_to_score_map = pickle.load(open(traj_to_score_path(hoc_num)))
    return traj_to_score_map


def find_missing_ast_ids(hoc_num):
    missing_ids = []
    program_embeddings_map = get_ast_id_to_program_embedding_map(hoc_num)
    ast_ids = sorted(program_embeddings_map.keys())
    max_ast_id = ast_ids[-1]
    for id in xrange(max_ast_id + 1):
        if id not in ast_ids:
            missing_ids.append(id)
    filename = '../preprocessed_data/hoc{}/missing_ast_ids.txt'.format(hoc_num)
    check_if_path_exists_or_create(filename)
    write_list_to_file(missing_ids, filename)
    return missing_ids


def get_missing_ast_ids(hoc_num):
    missing_ids = []
    filename = '../preprocessed_data/hoc{}/missing_ast_ids.txt'.format(hoc_num)
    with open(filename, 'rb') as f:
        missing_ids = [int(l) for l in f]
    return missing_ids


def write_list_to_file(l, filename):
    with open(filename, 'wb') as f:
        for r in l:
            f.write(str(r) + '\n')


def get_traj_counts(hoc_num):
    # returns a map from trajectory id to count
    traj_to_count_map = Counter()

    filename = '../data/hoc{}/trajectories/counts.txt'.format(hoc_num)
    with open(filename, 'rb') as f:
        for i, line in enumerate(f):
            if i == 0: # header
                continue
            traj_id, count = line.split()
            traj_id, count = int(traj_id.strip()), int(count.strip())
            traj_to_count_map[traj_id] = count

    return traj_to_count_map


def get_ast_counts(hoc_num):
    # returns a map from trajectory id to count
    ast_to_count_map = Counter()

    filename = '../data/hoc{}/asts/counts.txt'.format(hoc_num)
    with open(filename, 'rb') as f:
        for i, line in enumerate(f):
            if i == 0: # header
                continue
            ast_id, count = line.split()
            ast_id, count = int(ast_id.strip()), int(count.strip())
            ast_to_count_map[ast_id] = count

    return ast_to_count_map




if __name__ == "__main__":
    # m = get_traj_to_total_steps_map(18)
    # print m.items()[:100]
    # students = get_students_who_solved_next_problem(18)
    # print students[:10]

    # x = load_x_program_embeddings(18)
    # for i in xrange(100,110):
    #     print (len(x[i]))
    #     print (x[i])

    # all_student_ids = get_all_student_ids(18)
    #
    # print (len(all_student_ids))
    # print (all_student_ids[:10])

    # traj_id_to_embeddings_map = load_program_embeddings(18, max_timesteps=10)

    #
    # x, y = load_data_will_student_solve_next_problem(hoc_num=18, minlen=3)
    # print ("length of first sequence: {}".format(len(x[0])))
    # print ("first embedding vector: {}".format(x[0][0]))
    # print ("true y: {}".format(y[0]))


    print_all_asts_in_traj(18, 3)


    # x, y = load_data_will_student_solve_current_problem(18, minlen=3)
    # print x.shape
    # print x[0]


