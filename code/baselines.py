from utils import *
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from preprocess import *

from sklearn import linear_model


def baseline_pathscore(hoc_num, minlen=3, test_size=0.1):
    x, y = load_data_will_student_solve_next_problem_baseline(hoc_num, minlen=minlen)
    success_rate = np.average(y)
    print ("Success rate: {}".format(success_rate))
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size, random_state=42)
    logistic = linear_model.LogisticRegression()
    logistic.fit(x_train, y_train)
    pred_train = logistic.predict(x_train)
    train_acc = accuracy_score(y_train, pred_train)
    pred_test = logistic.predict(x_test)
    test_acc = accuracy_score(y_test, pred_test)
    print ("Baseline logistic regression on path scores:\n\n\
    Only including trajectories with minimum length: {}\n\
    Number of samples in dataset: {}\n\
    Test size: {}\n\
    Train acc: {}\t Test acc: {}\n\n".format(minlen, x.shape[0], test_size, train_acc, test_acc))
    return train_acc, test_acc



if __name__ == '__main__':
    hoc_num = 18
    baseline_pathscore(hoc_num, minlen=1)


