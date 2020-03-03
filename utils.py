import numpy as np


def unit_gaussian(x, mu, sigma):
    # inv_cov = sigma
    D = mu.shape[0]
    # print(sigma)
    exponent = np.exp((-0.5) * np.dot(np.dot((x - mu), sigma), (x - mu).T))
    # print(exponent)
    z = 1 / (((2 * np.pi) ** (D / 2)) * (np.abs(sigma) ** 0.5))
    # print(z * exponent)
    return z * exponent


def calculate_likelihood(N, K, data, mu_k, cov_k, pi_k):
    log_likelihood = 0
    print(K)
    for n in range(N):
        temp = 0
        for k in range(K):
            # temp += pi_k[k] * (multivariate_normal.pdf(data[n],mu_k[k],cov_k[k]))
            # print(pi_k[k])
            temp += pi_k[k] * (unit_gaussian(data[n], mu_k[k, :], cov_k[k, :]))
            # print(temp)
        log_likelihood += np.log(temp)
    return log_likelihood/N

# print(float('inf')/255)