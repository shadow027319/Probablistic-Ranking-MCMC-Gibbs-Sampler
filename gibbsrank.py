import scipy.linalg
import numpy as np
from tqdm import tqdm


def gibbs_sample(G, M, num_iters, random = False):
    # Number of games
    N = G.shape[0]

    # Array containing mean skills of each player, set to prior mean
    w = np.zeros((M, 1))
    if random:
        w = 0.1 * np.random.randn(M, 1)

    # Array that will contain skill samples
    skill_samples = np.zeros((M, num_iters))

    # Array containing skill variance for each player, set to prior variance
    pv = 0.5 * np.ones(M)

    # Number of iterations of Gibbs
    for i in tqdm(range(num_iters)):

        # Sample performance given differences in skills and outcomes
        t = np.zeros((N, 1))
        for g in range(N):

            s = w[G[g, 0]] - w[G[g, 1]]       # Difference in skills
            t[g] = s + np.random.randn()      # Sample performance
            while t[g] < 0:                   # Rejection step
                t[g] = s + np.random.randn()  # Resample if rejected

        # Jointly sample skills given performance differences
        # Compute the mean of the skills
        m = np.zeros((M, 1))
        for g in range(N):
            m[G[g, 0]] += t[g]
            m[G[g, 1]] -= t[g]

        #for p in range(M):
        #    m[p] = sum([t[g] * (p == G[g, 0] - p == G[g, 1]) for g in range(N)])

        # Container for sum of precision matrices (likelihood terms)
        iS = np.zeros((M, M))

        for g in range(N):
            # First cover the diagonal terms
            iS[G[g, 0], G[g, 0]] += 1
            iS[G[g, 1], G[g, 1]] += 1

            # Cover the non-diagonal terms
            iS[G[g, 0], G[g, 1]] -= 1
            iS[G[g, 1], G[g, 0]] -= 1

        # Posterior precision matrix
        iSS = iS + np.diag(1. / pv)

        # Use Cholesky decomposition to sample from a multivariate Gaussian
        iR = scipy.linalg.cho_factor(iSS)  # Cholesky decomposition of the posterior precision matrix
        mu = scipy.linalg.cho_solve(iR, m, check_finite=False)  # uses cholesky factor to compute inv(iSS) @ m

        # sample from N(mu, inv(iSS))
        w = mu + scipy.linalg.solve_triangular(iR[0], np.random.randn(M, 1), check_finite=False)
        skill_samples[:, i] = w[:, 0]
    return skill_samples


