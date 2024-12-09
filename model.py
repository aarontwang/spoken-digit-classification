import sys
import os
import numpy as np

import matplotlib as mpl
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture as GMM
from sklearn.mixture._gaussian_mixture import _compute_precision_cholesky

from scipy.stats import multivariate_normal

from data import *


class DigitClassifier:
    def __init__(self):
        self.n_features = 13

        self.train_data, self.train_labels = load_train_data()
        self.test_data, self.test_labels = load_test_data()

        self.train_predictions = []
        self.train_accuracy = None

        self.test_predictions = []
        self.test_accuracy = None

        self.cluster_labels_list = None

        self.gmm_param_list = []

        self.use_gender = False

    def get_gmm_list(self, kmeans: bool, covariance_type, n_components, use_gender=False):
        self.test_predictions = []
        self.train_predictions = []
        self.gmm_param_list = []
        self.use_gender = use_gender

        # Train GMM for each digit
        for digit in range(10):
            target_digits = self.train_data[self.train_labels == digit]
            data = np.vstack(target_digits.flatten())

            if use_gender:
                self.train_gmm(data[:len(data)//2], kmeans, covariance_type, n_components)
                self.train_gmm(data[len(data)//2:], kmeans, covariance_type, n_components)
            else:
                self.train_gmm(data, kmeans, covariance_type, n_components)

    def train_gmm(self, data, kmeans: bool, covariance_type, n_components):
        gmm = None

        if kmeans:
            cluster_weights, cluster_means, cluster_covariances, cluster_precisions, \
                cluster_precisions_cholesky = self.get_kmeans_gmm(data, covariance_type, n_components)

            # Create GMM by manually setting all of its parameters
            gmm = GMM(n_components=n_components, covariance_type='full')
            gmm.weights_ = cluster_weights
            gmm.covariances_ = cluster_covariances
            gmm.precisions_ = cluster_precisions
            gmm.means_ = cluster_means
            gmm.precisions_cholesky_ = cluster_precisions_cholesky
        else:
            gmm = GMM(n_components=n_components, covariance_type=covariance_type).fit(data)

        self.gmm_param_list.append(gmm)

    def get_kmeans_gmm(self, data, covariance_type, n_components):
        kmeans = KMeans(n_clusters=n_components, random_state=0, n_init="auto").fit(data)
        labels = kmeans.predict(data)

        cluster_weights = []
        cluster_means = []
        cluster_covariances = []
        cluster_precisions = []
        cluster_precisions_cholesky = []

        if covariance_type == 'tied' or covariance_type == 'tied_diag' or covariance_type == 'tied_spherical':
            data_tied = data.copy()

        for i in range(n_components):
            cluster_frames = data[labels == i]
            cluster_mean = np.mean(cluster_frames, axis=0, dtype=np.float64)

            # Calculate covariance
            if covariance_type == 'full':
                cluster_cov = np.cov(cluster_frames, rowvar=False)
            elif covariance_type == 'diag':
                cluster_cov = np.zeros((cluster_mean.shape[0], cluster_mean.shape[0]))
                for j in range(cluster_mean.shape[0]):
                    cluster_cov[j, j] = np.mean((cluster_frames[:, j] - cluster_mean[j]) ** 2)
            elif covariance_type == 'spherical':
                var = np.var((cluster_frames - cluster_mean).flatten())
                cluster_cov = np.identity(cluster_mean.shape[0]) * var
            elif covariance_type == 'tied' or covariance_type == 'tied_diag' or covariance_type == 'tied_spherical':
                data_tied[labels == i] = data_tied[labels == i] - cluster_mean

            # Calculate weight of the cluster
            cluster_weight = len(cluster_frames) / len(data)

            cluster_weights.append(cluster_weight)
            cluster_means.append(cluster_mean)
            if not (covariance_type == 'tied' or covariance_type == 'tied_diag' or covariance_type == 'tied_spherical'):
                cluster_covariances.append(cluster_cov)
                cluster_precisions.append(np.linalg.inv(cluster_cov))

        if covariance_type == 'tied':
            cluster_cov = np.cov(data_tied, rowvar=False)
            cluster_covariances = [cluster_cov] * n_components
        elif covariance_type == 'tied_diag':
            data_mean = np.mean(data_tied, axis=0, dtype=np.float64)
            cluster_cov = np.zeros((data_mean.shape[0], data_mean.shape[0]))

            for j in range(data_mean.shape[0]):
                cluster_cov[j, j] = np.mean((data_tied[:, j] - data_mean[j]) ** 2)

            cluster_covariances = [cluster_cov] * n_components
        elif covariance_type == 'tied_spherical':
            data_mean = np.mean(data_tied, axis=0, dtype=np.float64)
            var = np.var((data_tied - data_mean).flatten())
            cluster_cov = np.identity(cluster_mean.shape[0]) * var

            cluster_covariances = [cluster_cov] * n_components

        cluster_weights, cluster_means, cluster_covariances, cluster_precisions = np.asarray(cluster_weights), \
            np.asarray(cluster_means), \
            np.asarray(cluster_covariances), \
            np.asarray(cluster_precisions)

        cluster_precisions_cholesky = _compute_precision_cholesky(cluster_covariances, "full")

        return cluster_weights, cluster_means, cluster_covariances, cluster_precisions, cluster_precisions_cholesky

    def predict(self, test: bool):
        # check that GMM has been trained
        assert (len(self.gmm_param_list) > 0)

        if (test and len(self.test_predictions) == 0) or (not test and len(self.train_predictions) == 0):
            if test:
                self.test_predictions = np.zeros(self.test_labels.shape)
            else:
                self.train_predictions = np.zeros(self.train_labels.shape)

            data_range = len(self.test_data) if test else len(self.train_data)
            for i in range(data_range):
                likelihoods = []
                for digit in range(10):
                    if self.use_gender:
                        gmm_male = self.gmm_param_list[digit*2]
                        gmm_female = self.gmm_param_list[digit*2 + 1]

                        score = max(gmm_male.score(self.test_data[i]) if test else gmm_male.score(self.train_data[i]), 
                                    gmm_female.score(self.test_data[i]) if test else gmm_female.score(self.train_data[i]))
                    else:
                        gmm = self.gmm_param_list[digit]
                        score = gmm.score(self.test_data[i]) if test else gmm.score(self.train_data[i])

                    likelihoods.append(score)

                if test:
                    self.test_predictions[i] = np.argmax(np.array(likelihoods))
                else:
                    self.train_predictions[i] = np.argmax(np.array(likelihoods))

        if test:
            return self.test_predictions
        else:
            return self.train_predictions

    def evaluate(self, test: bool):
        true_labels = self.test_labels if test else self.train_labels
        acc = np.sum(np.where(self.predict(test) == true_labels, 1, 0)) / len(true_labels)

        if test:
            self.test_accuracy = acc
        else:
            self.train_accuracy = acc

        return acc
