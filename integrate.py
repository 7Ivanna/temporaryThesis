import numpy as np
from scipy.integrate import dblquad

# Gaussian function
def gaussian(t, A, mu, sigma):
    return A * np.exp(-((t - mu)**2) / (2 * sigma**2))

# i think i do this....
def alpha_1(t1, t2):
    A1, mu1, sigma1 = 1, 0, 1
    return gaussian(t1, A1, mu1, sigma1) * gaussian(t2, A1, mu1, sigma1)

def alpha_2(t1, t2):
    A2, mu2, sigma2 = 1, 0, 1
    return gaussian(t1, A2, mu2, sigma2) * gaussian(t2, A2, mu2, sigma2)

#  integrals
integral_1 = dblquad(alpha_1, -np.inf, np.inf, lambda x: -np.inf, lambda x: np.inf)
integral_2 = dblquad(alpha_2, -np.inf, np.inf, lambda x: -np.inf, lambda x: np.inf)

# integrals into a column vector ... seems like cheating....
psi = np.array([
    [integral_1[0]],
    [integral_2[0]]
])

print("The state psi is:")
print(psi)
