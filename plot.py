import sys
import os
import numpy as np

import matplotlib as mpl
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture as GMM
from sklearn.mixture._gaussian_mixture import _compute_precision_cholesky
from sklearn.metrics import ConfusionMatrixDisplay

from scipy.stats import multivariate_normal
from sklearn.neighbors import KernelDensity

from model import *
from data import *


def plot_pdf(savepath, classifier, kmeans: bool, covariance_type, mfcc_pairs):
    for digit in range(10):
        gmm = classifier.gmm_param_list[digit]
        n_components = gmm.n_components
        test_data = np.vstack(classifier.test_data[classifier.test_labels == digit].flatten())

        fig, axs = plt.subplots(1, len(mfcc_pairs), figsize=(8 * len(mfcc_pairs), 5))
        for j in range(len(mfcc_pairs)):
            x, y = mfcc_pairs[j]
            x_1 = np.linspace(np.min(test_data[:, x]) - 1, np.max(test_data[:, x]) + 1, 200)
            x_2 = np.linspace(np.min(test_data[:, y]) - 1, np.max(test_data[:, y]) + 1, 200)
            x_1_grid, x_2_grid = np.meshgrid(x_1, x_2)
            pos = np.dstack((x_1_grid, x_2_grid))

            # plot the contours
            axs[j].scatter(test_data[:, x], test_data[:, y], s=10, label='Data Points')

            for k in range(n_components):
                # calculate the GMM pdf over the grid
                if kmeans:
                    pdf = multivariate_normal.pdf(x=pos, mean=gmm.means_[k][[x, y]],
                                                  cov=gmm.covariances_[k][[x, y], :][:, [x, y]])
                else:
                    if covariance_type == 'full':
                        pdf = multivariate_normal.pdf(x=pos, mean=gmm.means_[k][[x, y]],
                                                      cov=gmm.covariances_[k][[x, y], :][:, [x, y]])
                    elif covariance_type == 'diag':
                        pdf = multivariate_normal.pdf(x=pos, mean=gmm.means_[k][[x, y]],
                                                      cov=np.diag(gmm.covariances_[k])[[x, y], :][:, [x, y]])
                    elif covariance_type == 'tied':
                        pdf = multivariate_normal.pdf(x=pos, mean=gmm.means_[k][[x, y]],
                                                      cov=gmm.covariances_[[x, y], :][:, [x, y]])
                    elif covariance_type == 'tied_diag':
                        pdf = multivariate_normal.pdf(x=pos, mean=gmm.means_[k][[x, y]],
                                                      cov=np.diag(gmm.covariances_)[[x, y], :][:, [x, y]])
                axs[j].contour(x_1_grid, x_2_grid, pdf, levels=20, cmap="viridis")

            axs[j].set_xlabel("MFCC_{}".format(x + 1))
            axs[j].set_ylabel("MFCC_{}".format(y + 1))
            axs[j].set_title("MFCC {} vs MFCC {}".format(x + 1, y + 1))

        fig.suptitle("GMM with {} PDF Contour Plot for Digit {}".format("K-Means" if kmeans else "EM", digit))
        plt.tight_layout()
        plt.savefig("{}_{}.jpg".format(savepath, digit), bbox_inches='tight', pad_inches=0)
        plt.close()


def plot_confusion_matrix(savepath, labels, predictions, title):
    plt.figure(figsize=(10, 10))
    plt.rcParams.update({'font.size': 8})
    cm = ConfusionMatrixDisplay.from_predictions(labels, predictions, normalize='true', cmap='Blues', values_format=".3f")
    ax = cm.ax_
    ax.set_xlabel('Predicted Label', fontdict={'size': '10'})
    ax.set_ylabel('True Label', fontdict={'size': '10'})
    ax.set_title(title, fontdict={'size': '12'})
    plt.tight_layout()
    plt.savefig("{}.jpg".format(savepath), bbox_inches='tight', pad_inches=0)
    plt.close()


def plot_contours():
    mfcc_pairs = [[0, 1], [0, 2], [1, 2]]

    classifier = DigitClassifier()

    classifier.get_gmm_list(kmeans=True, covariance_type='diag', n_components=10)
    classifier.predict(test=True)
    plot_pdf("figures/pdf_contour_kmeans_diag_", classifier, True, 'diag', mfcc_pairs)

    plot_confusion_matrix("figures/confusion_matrix_diag_kmeans", classifier.test_labels, classifier.test_predictions,
                          "Confusion Matrix for GMM with K-Means and Diagonal Covariance")

    classifier.get_gmm_list(kmeans=False, covariance_type='diag', n_components=10)
    classifier.predict(test=True)
    plot_pdf("figures/pdf_contour_em_diag_", classifier, False, 'diag', mfcc_pairs)

    plot_confusion_matrix("figures/confusion_matrix_diag_em", classifier.test_labels, classifier.test_predictions,
                          "Confusion Matrix for GMM with EM and Diagonal Covariance")

    classifier.get_gmm_list(kmeans=True, covariance_type='tied', n_components=10)
    classifier.predict(test=True)
    plot_pdf("figures/pdf_contour_kmeans_tied_", classifier, True, 'tied', mfcc_pairs)

    plot_confusion_matrix("figures/confusion_matrix_tied_kmeans", classifier.test_labels, classifier.test_predictions,
                          "Confusion Matrix for GMM with K-Means and Tied Covariance")

    classifier.get_gmm_list(kmeans=False, covariance_type='tied', n_components=10)
    classifier.predict(test=True)
    plot_pdf("figures/pdf_contour_em_tied_", classifier, False, 'tied', mfcc_pairs)

    plot_confusion_matrix("figures/confusion_matrix_tied_em", classifier.test_labels, classifier.test_predictions,
                          "Confusion Matrix for GMM with EM and Tied Covariance")

    classifier.get_gmm_list(kmeans=True, covariance_type='tied_diag', n_components=10)
    classifier.predict(test=True)
    plot_pdf("figures/pdf_contour_kmeans_tied_diag_", classifier, True, 'tied_diag', mfcc_pairs)

    plot_confusion_matrix("figures/confusion_matrix_tied_diag_kmeans", classifier.test_labels, classifier.test_predictions,
                          "Confusion Matrix for GMM with K-Means and Tied Diagonal Covariance")

    classifier.get_gmm_list(kmeans=False, covariance_type='tied_diag', n_components=10)
    classifier.predict(test=True)
    plot_pdf("figures/pdf_contour_em_tied_diag_", classifier, False, 'tied_diag', mfcc_pairs)

    plot_confusion_matrix("figures/confusion_matrix_tied_diag_em", classifier.test_labels, classifier.test_predictions,
                          "Confusion Matrix for GMM with EM and Tied Diagonal Covariance")


def plot_mfccs(savepath):
    classifier = DigitClassifier()
    data = classifier.train_data

    for digit in range(10):
        x = np.arange(0, len(data[classifier.train_labels == digit][0]))

        plt.figure(figsize=(8, 5))
        for i in range(13):
            mfcc = data[classifier.train_labels == digit][0][:, i]
            plt.plot(x, mfcc, label="MFCC{}".format(i))

        plt.title("Digit {}".format(digit))
        plt.xlabel("Frame Index")
        plt.ylabel("MFCCs")
        plt.legend(loc='best', ncol=4)
        plt.savefig("{}_{}.jpg".format(savepath, digit), bbox_inches='tight')
        plt.close()


def scatter_plot_mfccs(savepath):
    classifier = DigitClassifier()
    data = classifier.train_data

    for digit in range(10):
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
        fig.set_size_inches(15, 5)
        fig.suptitle("Digit {}".format(digit))

        mfcc1 = data[classifier.train_labels == digit][0][:, 0]
        mfcc2 = data[classifier.train_labels == digit][0][:, 1]
        mfcc3 = data[classifier.train_labels == digit][0][:, 2]

        ax1.scatter(mfcc1, mfcc2, label='MFCC1 vs MFCC2')
        ax1.set_title('MFCC1 vs MFCC2')
        ax1.set_xlabel('MFCC1')
        ax1.set_ylabel('MFCC2')

        ax2.scatter(mfcc1, mfcc3, label='MFCC1 vs MFCC2')
        ax2.set_title('MFCC1 vs MFCC3')
        ax2.set_xlabel('MFCC1')
        ax2.set_ylabel('MFCC3')

        ax3.scatter(mfcc2, mfcc3, label='MFCC1 vs MFCC2')
        ax3.set_title('MFCC2 vs MFCC3')
        ax3.set_xlabel('MFCC2')
        ax3.set_ylabel('MFCC3')

        plt.tight_layout()
        plt.savefig("{}_{}.jpg".format(savepath, digit), bbox_inches='tight')
        plt.close()


if __name__ == '__main__':
    plot_contours()
    plot_mfccs('figures/mfcc_pairs')
    scatter_plot_mfccs("figures/mfcc_scatter")
