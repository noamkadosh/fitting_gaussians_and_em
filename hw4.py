import numpy as np
import pandas as pd
from scipy.stats import norm
import matplotlib.pyplot as plt

PLOT_WIDTH_IN_SIGMA = 3


# code for ploting the original distribution with the one that we found with the EM
def plot_pred_vs_actual(df):
    plt.figure(figsize=(10, 7))
    mu_hat1 = df['x'][df['z'] == 0].mean()
    sigma_hat1 = df['x'][df['z'] == 0].std()
    x_hat1 = np.linspace(mu_hat1 - PLOT_WIDTH_IN_SIGMA * sigma_hat1, mu_hat1 + PLOT_WIDTH_IN_SIGMA * sigma_hat1, 1000)
    y_hat1 = norm.pdf(x_hat1, mu_hat1, sigma_hat1)

    mu_hat2 = df['x'][df['z'] == 1].mean()
    sigma_hat2 = df['x'][df['z'] == 1].std()
    x_hat2 = np.linspace(mu_hat2 - PLOT_WIDTH_IN_SIGMA * sigma_hat2, mu_hat2 + PLOT_WIDTH_IN_SIGMA * sigma_hat2, 1000)
    y_hat2 = norm.pdf(x_hat2, mu_hat2, sigma_hat2)

    plt.plot(x_hat1, y_hat1, color='red', lw=1, ls='-', alpha=0.5)
    plt.plot(x_hat2, y_hat2, color='blue', lw=1, ls='-', alpha=0.5)

    plt.xlim(min(mu_hat1 - PLOT_WIDTH_IN_SIGMA * sigma_hat1, mu_hat2 - 3 * sigma_hat2),
             max(mu_hat1 + PLOT_WIDTH_IN_SIGMA * sigma_hat1, mu_hat2 + PLOT_WIDTH_IN_SIGMA * sigma_hat2))

    mu1 = -1
    sigma1 = 1
    x1 = np.linspace(mu1 - PLOT_WIDTH_IN_SIGMA * sigma1, mu1 + PLOT_WIDTH_IN_SIGMA * sigma1, 1000)
    y1 = norm.pdf(x1, mu1, sigma1)

    mu2 = 5
    sigma2 = 2
    x2 = np.linspace(mu2 - PLOT_WIDTH_IN_SIGMA * sigma2, mu2 + PLOT_WIDTH_IN_SIGMA * sigma2, 1000)
    y2 = norm.pdf(x2, mu2, sigma2)

    plt.plot(x1, y1, color='red', lw=1, ls='--', alpha=0.5)
    plt.plot(x2, y2, color='blue', lw=1, ls='--', alpha=0.5)

    plt.xlim(min(mu1 - PLOT_WIDTH_IN_SIGMA * sigma1, mu2 - PLOT_WIDTH_IN_SIGMA * sigma2),
             max(mu1 + PLOT_WIDTH_IN_SIGMA * sigma1, mu2 + PLOT_WIDTH_IN_SIGMA * sigma2))

    plt.legend(['Predicted - 1st gaussian', 'Predicted - 2nd gaussian',
                'Original - 1st gaussian', 'Original - 2nd gaussian'])

    plt.show()
    print ("mu_1: %s ,predicted mu_1: %s\nsigma_1: %s, predicted sigma_1: %s" % (mu1, mu_hat1, sigma1, sigma_hat1))
    print ("mu_2: %s ,predicted mu_2: %s\nsigma_2: %s, predicted sigma_2: %s" % (mu2, mu_hat2, sigma2, sigma_hat2))


def get_num_of_gaussians():
    k = 4

    return k


def init(points_list, k):
    """
    :param points_list: the entire data set of points. type: list.
    :param k: number of gaussians. type: integer.
    :return the initial guess of w, mu, sigma. types: array
    """
    mu = np.array([])
    size = np.subtract(points_list[-1], points_list[0])
    sigma = np.empty(k)
    sigma.fill(np.divide(size, k))
    weight = np.divide(1, k)
    w = np.empty(k)
    w.fill(weight)
    points_list.sort()
    mu_interval = np.divide(size, k)
    for i in range(k):
        current_mu = np.add(points_list[0], np.multiply(mu_interval, i))
        mu = np.append(mu, current_mu)



    return w, mu, sigma


def expectation(points_list, mu, sigma, w):
    """
    :param points_list: the entire data set of points. type: list.
    :param mu: expectation of each gaussian. type: array
    :param sigma: std for of gaussian. type: array
    :param w: weight of each gaussian. type: array
    :return likelihood: dividend of ranks matrix (likelihood). likelihood[i][j] is the likelihood of point i to belong to gaussian j. type: array
    """
    likelihood = np.zeros(shape=(len(points_list), np.size(mu)))
    for i in range(len(points_list)):
        for j in range(np.size(mu)):
            likelihood[i][j] = np.multiply(w[j], calc_normal_distribution(points_list[i], mu[j], sigma[j]))

    return likelihood


def calc_normal_distribution(point, mu, sigma):
    first_factor = np.divide(1, np.multiply(sigma, np.sqrt(np.multiply(2, np.pi))))
    power_of_e = np.divide(
        -np.power(np.subtract(point, mu), 2),
        np.multiply(2, np.power(sigma, 2))
    )
    e = np.power(np.e, power_of_e)
    distribution = np.multiply(first_factor, e)

    return distribution


def maximization(points_list, ranks):
    """
    # :param data: the complete data
    :param points_list: the entire data set of points. type: list.

    :param ranks: ranks matrix- r(x,k)- responsibility of each data point x to gaussian k
    :return w_new: new weight parameter of each gaussian
            mu_new: new expectation parameter of each gaussian
            sigma_new: new std parameter of each gaussian
    """
    stacked_points_list = np.column_stack((points_list, points_list))
    if len(ranks[0]) > 2:
        for i in range(len(ranks[0]) - 2):
            stacked_points_list = np.column_stack((stacked_points_list, points_list))
    w_new = np.multiply(np.divide(1, len(points_list)), np.sum(ranks, axis=0))
    mu_new = np.multiply(
        np.divide(1, np.multiply(w_new, len(points_list))),
        np.sum(np.multiply(ranks, stacked_points_list), axis=0))
    sigma_new = np.sqrt(np.multiply(
        np.divide(1, np.multiply(w_new, len(points_list))),
        np.sum(np.multiply(ranks, np.power(np.subtract(stacked_points_list, mu_new), 2)), axis=0)))

    return w_new, mu_new, sigma_new


def calc_max_delta(old_param, new_param):
    """
    :param old_param: old parameters to compare
    :param new_param: new parameters to compare
    :return maximal delta between each old and new parameter
    """

    max_delta = np.mean(np.subtract(np.abs(old_param), np.abs(new_param)))

    return max_delta


# helper function for plotting
def plot_gmm(k, res, mu, sigma, points_list, iter_num=-1):
    data = pd.DataFrame(points_list, columns=['x'])
    res = pd.DataFrame(res, columns=['x'])
    for k in range(k):
        res_bin = res[res == k]
        dots = data["x"][res_bin.index]
        plt.scatter(dots.values, norm.pdf(dots.values, loc=mu[k], scale=sigma[k]),
                    label="mu=%.2f, Sigma=%.2f" % (mu[k], sigma[k]), s=10)
    plt.ylabel('probability')
    if iter_num >= 0:
        plt.title('Expectation Maximization - GMM - iteration {}'.format(iter_num))
    else:
        plt.title('Expectation Maximization - GMM')
    plt.legend()
    plt.ylim(0, 0.5)
    plt.show()


def expectation_maximization(points_list, k, max_iter, epsilon):
    """
    :param points_list: the entire data set of points. type: list.
    :param k: number of gaussians. type: integer
    :param max_iter: maximal number of iterations to perform. type: integer
    :param epsilon: minimal change in parameters to declare convergence. type: float
    :return res: gaussian estimation for each point. res[i] is the gaussian number of the i-th point. type: list
            mu: mu values of each gaussian. type: array
            sigma: sigma values of each gaussian. type: array
            log_likelihood: a list of the log likelihood values each iteration. type: list


    """
    w, mu, sigma = init(points_list, k)
    # Loop until convergence
    delta = np.infty
    iter_num = 0

    log_likelihood = []
    while delta > epsilon and iter_num <= max_iter:

        # E step
        likelihood = expectation(points_list, mu, sigma, w)

        likelihood_sum = likelihood.sum(axis=1)
        log_likelihood.append(np.sum(np.log(likelihood_sum), axis=0))

        # M step

        ranks = np.divide(likelihood, likelihood.sum(axis=0))

        w_new, mu_new, sigma_new = maximization(points_list, ranks)

        # Check significant change in parameters
        delta = max(calc_max_delta(w, w_new), calc_max_delta(mu, mu_new), calc_max_delta(sigma, sigma_new))

        w = w_new
        mu = mu_new
        sigma = sigma_new

        if iter_num % 10 == 0:
            res = ranks.argmax(axis=1)
            plot_gmm(k, res, mu, sigma, points_list, iter_num)
        iter_num += 1

    plt.show()

    res = ranks.argmax(axis=1)

    # Display estimated Gaussian:
    plot_gmm(k, res, mu, sigma, points_list, iter_num)

    return res, mu, sigma, log_likelihood
