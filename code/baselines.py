from utils import *
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from preprocess import *

from sklearn import linear_model
from sklearn.svm import SVC



def baseline_pathscore(hoc_num, minlen=3, maxlen=10, test_size=0.1, use_pathscore_only=True, model='log_reg'):
    if use_pathscore_only:
        x, y = load_data_will_student_solve_next_problem_baseline(hoc_num, minlen=minlen)
    else:
        x, y = load_data_will_student_solve_next_problem_baseline_pathscore_success_on_cur_problem(hoc_num, minlen=minlen)
    print ("x shape: {}".format(x.shape))
    success_rate = np.average(y)
    print ("Success rate: {}".format(success_rate))

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size, random_state=42)

    clf = None
    if model == 'log_reg':
        clf = linear_model.LogisticRegression()
    elif model == 'svm':
        clf = SVC()
    clf.fit(x_train, y_train)
    pred_train = clf.predict(x_train)
    train_acc = accuracy_score(y_train, pred_train)
    pred_test = clf.predict(x_test)
    test_acc = accuracy_score(y_test, pred_test)
    print ("Baseline logistic regression on path scores:\n\n\
    Only including trajectories with minimum length: {}\n\
    Number of samples in dataset: {}\n\
    Test size: {}\n\
    Train acc: {}\t Test acc: {}\n\n".format(minlen, x.shape[0], test_size, train_acc, test_acc))
    return train_acc, test_acc



def baseline_pathscore_traj_len(hoc_num, only_traj_len=5, test_size=0.1, model='log_reg', num_trials=1):
    x, y =load_data_will_student_solve_next_problem_baseline_pathscore_success_on_cur_problem_traj_len(hoc_num, only_traj_len=only_traj_len)
    print ("x shape: {}".format(x.shape))
    n_samples = x.shape[0]
    success_rate = np.average(y)
    print ("Success rate: {}".format(success_rate))


    train_accs, test_accs = [], []
    for i in xrange(num_trials):
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size)


        clf = None
        if model == 'log_reg':
            clf = linear_model.LogisticRegression()
        elif model == 'svm':
            clf = SVC()
        clf.fit(x_train, y_train)
        pred_train = clf.predict(x_train)
        train_acc = accuracy_score(y_train, pred_train)
        pred_test = clf.predict(x_test)
        test_acc = accuracy_score(y_test, pred_test)
        print ("Baseline logistic regression on path scores:\n\n\
        Only including trajectories with length: {}\n\
        Number of samples in dataset: {}\n\
        Test size: {}\n\
        Train acc: {}\t Test acc: {}\n\n".format(only_traj_len, n_samples, test_size, train_acc, test_acc))
        train_accs.append(train_acc)
        test_accs.append(test_acc)
    return train_accs, test_accs, n_samples





if __name__ == '__main__':
    hoc_num = 18
    baseline_pathscore(hoc_num, minlen=1)


