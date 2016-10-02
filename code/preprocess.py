# data_utils.py
# @author: Lisa Wang
# @created: Oct 1 2016
#
#===============================================================================
# DESCRIPTION:
#
# This module creates the student id to trajectory id map and the trajectory id
# to asts map for a specified hoc number.
#
#===============================================================================
# CURRENT STATUS: Working
#===============================================================================
# USAGE: python preprocess.py -n <hoc_num>
# or to get help: python preprocess.py -h
#

import os, sys, getopt
import pickle
import csv
from collections import defaultdict


def get_student_to_traj_map(hoc_num):
    student_to_traj_map = defaultdict()
    traj_dir_path = '../data/hoc{}/trajectories/'.format(hoc_num)
    if os.path.exists(traj_dir_path):
        with open(os.path.join(traj_dir_path, 'idMap.txt')) as f:
            reader = csv.reader(f)
            headers = reader.next()
            for row in reader:
                student_id, traj_id = int(row[0]), int(row[1])
                student_to_traj_map[student_id] = traj_id
    return student_to_traj_map


def get_traj_to_asts_map(hoc_num):
    traj_to_asts_map = defaultdict(list)
    traj_dir_path = '../data/hoc{}/trajectories/'.format(hoc_num)
    if os.path.exists(traj_dir_path):
        traj_files = [f for f in os.listdir(traj_dir_path) if (os.path.isfile(os.path.join(traj_dir_path, f)) and f not in ['counts.txt', 'idMap.txt'])]

        for f_path in traj_files:
            with open(os.path.join(traj_dir_path, f_path)) as f:
                asts = [int(l) for l in f]
                traj_id = int(os.path.splitext(f_path)[0])

                traj_to_asts_map[traj_id] = asts
    return traj_to_asts_map


def check_if_exists_or_create_file(file_path):
    if not os.path.exists(os.path.dirname(file_path)):
        try:
            os.makedirs(os.path.dirname(file_path))
        except OSError as exc:  # Guard against race condition
            return False
    return True


def create_maps(hoc_num):
    student_to_traj_map = get_student_to_traj_map(hoc_num)
    filename = '../preprocessed_data/hoc{}/student_id_to_trajectory_id_map.pickle'.format(hoc_num)
    check_if_exists_or_create_file(filename)
    pickle.dump(student_to_traj_map, open(filename, 'wb'))

    traj_to_asts_map = get_traj_to_asts_map(hoc_num)
    filename = '../preprocessed_data/hoc{}/trajectory_id_to_asts_map.pickle'.format(hoc_num)
    check_if_exists_or_create_file(filename)
    pickle.dump(traj_to_asts_map,
                open(filename, 'wb'))


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
    # for hoc_num in [4, 18]:
    #     create_maps(hoc_num)
    main()





