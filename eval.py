import sys
import os
import numpy as np

import matplotlib as mpl
import matplotlib.pyplot as plt

from model import *
from data import *


def test_hyperparams():
    covariance_types = ['full', 'tied', 'diag', 'tied_diag', 'spherical', 'tied_spherical']
    n_components = 10
    seeds = 10

    acc_kmeans = []
    for covariance_type in covariance_types:
        print("Training GMM K-Means with {} Covariance".format(covariance_type))
        test_acc = []
        for i in range(seeds):
            classifier = DigitClassifier()
            classifier.get_gmm_list(kmeans=True, covariance_type=covariance_type, n_components=n_components)
            test_acc.append(classifier.evaluate(True))

        acc_kmeans.append(np.mean(np.array(test_acc)))

        print("GMM K-Means, Covariance Type: {}, Mean Accuracy: {}".format(covariance_type, acc_kmeans[-1]))

    acc_em = []
    for covariance_type in covariance_types:
        test_acc = []
        for i in range(seeds):
            classifier = DigitClassifier()
            classifier.get_gmm_list(kmeans=False, covariance_type=covariance_type, n_components=n_components)
            test_acc.append(classifier.evaluate(True))

        acc_em.append(np.mean(np.array(test_acc)))

        print("GMM EM, Covariance Type: {}, Mean Accuracy: {}".format(covariance_type, acc_em[-1]))
    
    acc_em_use_gender = []
    for covariance_type in covariance_types:
        test_acc_gender = []
        for i in range(seeds):
            classifier = DigitClassifier()
            classifier.get_gmm_list(kmeans=False, covariance_type=covariance_type, n_components=n_components, use_gender=True)
            test_acc_gender.append(classifier.evaluate(True))

        acc_em_use_gender.append(np.mean(np.array(test_acc_gender)))

        print("GMM EM with Gender, Covariance Type: {}, Mean Accuracy: {}".format(covariance_type, test_acc_gender[-1]))
    


if __name__ == '__main__':
    test_hyperparams()
