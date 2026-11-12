import numpy as np
import random
from scipy.stats import gamma, lognorm, norm
import matplotlib.pyplot as plt
import math
import os
from scipy import stats


# -------------------------------
# One Dimensional Indices
# -------------------------------

def l2l1(values):
    """
    Computes the ratio ||x||_2 / ||x||_1 for a vector of positive values.

    Parameters
    ----------
    values : list or iterable of positive numbers

    Returns
    -------
    ratio : float
    """
    # if not values:
    #     raise ValueError("Input vector must not be empty.")
    # print(values)
    l1 = sum(values)
    if l1 == 0:
        raise ValueError("L1 norm is zero; ratio is undefined.")

    l2 = math.sqrt(sum(x * x for x in values))

    return l2 / l1


def gini(x):
    """
    Compute the Gini index of a 1D array of non-negative values.

    Parameters
    ----------
    x : array-like
        Non-negative values (must have positive mean).

    Returns
    -------
    g : float
        Gini index in [0, 1].
    """
    x = np.asarray(x, dtype=float)

    if np.any(x < 0):
        raise ValueError("Gini index is only defined for non-negative values.")

    if np.all(x == 0):
        return 0.0

    x = np.sort(x)
    n = x.size
    cumx = np.cumsum(x)

    g = (n + 1 - 2 * np.sum(cumx) / cumx[-1]) / n
    return g


def sinf(values):
    """
    Parameters
    ----------
    values : list or iterable of positive numbers

    Returns
    -------
    average : float
    sum_smallest_half : float
    sum_largest_half : float
    """
    # if not values:
    #     raise ValueError("Input vector must not be empty.")

    N = len(values)
    half = N // 2

    sorted_vals = sorted(values)

    average = sum(sorted_vals) / N
    sum_smallest_half = sum(sorted_vals[:half])
    sum_largest_half = sum(sorted_vals[-half:])

    return (sum_largest_half - sum_smallest_half)/(N*average)


def pietra(values):
    """
    Parameters
    ----------
    values : list or iterable of positive numbers

    Returns
    -------
    average : float
    count_below_1_over_N : int
    sum_below_1_over_N : float
    """
    # if not values:
    #     raise ValueError("Input vector must not be empty.")

    N = len(values)
    
    threshold = 1.0 / N
    
    # print(values[0])
    # print(threshold)

    average = sum(values) / N

    below = [x for x in values if x < average]
    count_below = len(below)
    sum_below = sum(below)

    return count_below/N-sum_below/(N*average)


# -------------------------------
# Sampling Techniques
# -------------------------------


def sample_quadratic(n_samples=1):
    """
    Samples from the PDF f(x) = 12 (x - 1/2)^2 on [0,1].

    Parameters
    ----------
    n_samples : int
        Number of samples

    Returns
    -------
    samples : list of float
    """
    samples = []

    while len(samples) < n_samples:
        x = random.random()          # Uniform in [0,1]
        y = random.random() * 3.0    # Upper bound of the PDF

        if y <= 12 * (x - 0.5) ** 2:
            samples.append(x)

    return samples


def sample_X(n, p1, p2, p3, seed=None):
    """
    Sample n points from the discrete random variable X.

    Parameters
    ----------
    n : int
        Number of samples.
    p1, p2, p3 : float
        Probabilities for (0,0), (10,0), (0,10).
        Must satisfy p1 > 0, p2 > 0, p3 > 0,
        and p1 + p2 + p3 < 1.
    seed : int or None
        Random seed for reproducibility.

    Returns
    -------
    samples : ndarray of shape (n, 2)
        Sampled points.
    """
    if not (p1 > 0 and p2 > 0 and p3 > 0):
        raise ValueError("p1, p2, p3 must be positive.")
    if p1 + p2 + p3 >= 1:
        raise ValueError("p1 + p2 + p3 must be < 1.")

    rng = np.random.default_rng(seed)

    points = np.array([
        [0, 0],
        [10, 0],
        [0, 10],
        [10, 10]
    ])

    probs = np.array([
        p1,
        p2,
        p3,
        1 - p1 - p2 - p3
    ])

    idx = rng.choice(4, size=n, p=probs)
    return points[idx]


def sample_dp(n_samples,
                         alpha1=2.0, beta1=6.0,
                         alpha2=6.0, beta2=2.0,
                         weight=0.5):
    """
    Samples from a bimodal distribution on [0,1]
    using a mixture of two Beta distributions.

    Parameters
    ----------
    n_samples : int
    alpha1, beta1 : float
        Shape parameters of first Beta
    alpha2, beta2 : float
        Shape parameters of second Beta
    weight : float in (0,1)
        Mixing probability for first component

    Returns
    -------
    samples : ndarray
    """
    u = np.random.rand(n_samples)
    samples = np.empty(n_samples)

    mask = u < weight
    samples[mask] = np.random.beta(alpha1, beta1, mask.sum())
    samples[~mask] = np.random.beta(alpha2, beta2, (~mask).sum())

    return samples[0]

# ------------------------------
# Sliced Double-Peaked
# ------------------------------


def Sliced_dp_Gini(X):
    M_sample = 2000
    res=0
    N= len(X)
    
    for k in range(M_sample):
        theta = 0.5*sample_dp(1)*np.pi
        v_1=np.cos(theta)
        v_2=np.sin(theta)
        
        G=np.dot(X,(v_1,v_2))
        
        res=gini(G)*(1/(k+1))+res*((k)/(k+1))
        
    return res


def Sliced_dp_l2l1(X):
    M_sample = 2000
    res=0
    N= len(X)
    
    for k in range(M_sample):
        theta = 0.5*sample_dp(1)*np.pi
        v_1=np.cos(theta)
        v_2=np.sin(theta)
        
        G=np.dot(X,(v_1,v_2))
        
        res=l2l1(G)*(1/(k+1))+res*((k)/(k+1))
        
    return res


def Sliced_dp_Pietra(X):
    M_sample = 2000
    res=0
    N= len(X)
    
    for k in range(M_sample):
        theta = 0.5*sample_dp(1)*np.pi
        v_1=np.cos(theta)
        v_2=np.sin(theta)
        
        G=np.dot(X,(v_1,v_2))
        
        res=pietra(G)*(1/(k+1))+res*((k)/(k+1))
        
    return res

def Sliced_dp_Sinf(X):
    M_sample = 2000
    res=0
    N= len(X)
    
    for k in range(M_sample):
        theta = 0.5*sample_dp(1)*np.pi
        v_1=np.cos(theta)
        v_2=np.sin(theta)
        
        G=np.dot(X,(v_1,v_2))
        
        res=sinf(G)*(1/(k+1))+res*((k)/(k+1))
        
    return res





# ------------------------------
# Sliced Quadratic
# ------------------------------

def Sliced_quad_Gini(X):
    M_sample = 2000
    res=0
    N= len(X)
    
    for k in range(M_sample):
        theta = 0.5*sample_quadratic(1)[0]*np.pi
        v_1=np.cos(theta)
        v_2=np.sin(theta)
        
        G=np.dot(X,(v_1,v_2))
        
        res=gini(G)*(1/(k+1))+res*((k)/(k+1))
        
    return res

def Sliced_quad_l2l1(X):
    M_sample = 2000
    res=0
    N= len(X)
    
    for k in range(M_sample):
        theta = 0.5*sample_quadratic(1)[0]*np.pi
        v_1=np.cos(theta)
        v_2=np.sin(theta)
        
        G=np.dot(X,(v_1,v_2))
        
        res=l2l1(G)*(1/(k+1))+res*((k)/(k+1))
        
    return res


def Sliced_quad_Pietra(X):
    M_sample = 2000
    res=0
    N= len(X)
    
    for k in range(M_sample):
        theta = 0.5*sample_quadratic(1)[0]*np.pi
        v_1=np.cos(theta)
        v_2=np.sin(theta)
        
        G=np.dot(X,(v_1,v_2))
        # print(G)
        res=pietra(G)*(1/(k+1))+res*((k)/(k+1))
        
        # print(res)
        
    return res

def Sliced_quad_Sinf(X):
    M_sample = 2000
    res=0
    N= len(X)
    
    for k in range(M_sample):
        theta = 0.5*sample_quadratic(1)[0]*np.pi
        v_1=np.cos(theta)
        v_2=np.sin(theta)
        
        G=np.dot(X,(v_1,v_2))
        
        res=sinf(G)*(1/(k+1))+res*((k)/(k+1))
        
    return res



# ------------------------------
# Sliced Uniform
# ------------------------------

def Sliced_unif_Gini(X):
    M_sample = 2000
    res=0
    N= len(X)
    
    for k in range(M_sample):
        theta = np.random.uniform(0.0, 0.5*np.pi)
        v_1=np.cos(theta)
        v_2=np.sin(theta)
        
        G=np.dot(X,(v_1,v_2))
        
        res=gini(G)*(1/(k+1))+res*((k)/(k+1))
        
    return res

def Sliced_unif_l2l1(X):
    M_sample = 2000
    res=0
    N= len(X)
    
    for k in range(M_sample):
        theta = np.random.uniform(0.0, 0.5*np.pi)
        v_1=np.cos(theta)
        v_2=np.sin(theta)
        
        G=np.dot(X,(v_1,v_2))
        
        res=l2l1(G)*(1/(k+1))+res*((k)/(k+1))
        
    return res

def Sliced_unif_Pietra(X):
    M_sample = 2000
    res=0
    N= len(X)
    
    for k in range(M_sample):
        theta = np.random.uniform(0.0, 0.5*np.pi)
        v_1=np.cos(theta)
        v_2=np.sin(theta)
        
        G=np.dot(X,(v_1,v_2))
        
        res=pietra(G)*(1/(k+1))+res*((k)/(k+1))
        
    return res

def Sliced_unif_Sinf(X):
    M_sample = 2000
    res=0
    N= len(X)
    
    for k in range(M_sample):
        theta = np.random.uniform(0.0, 0.5*np.pi)
        v_1=np.cos(theta)
        v_2=np.sin(theta)
        
        G=np.dot(X,(v_1,v_2))
        
        res=sinf(G)*(1/(k+1))+res*((k)/(k+1))
        
    return res


# ------------------------------
# Sliced Beta
# ------------------------------



def Sliced_beta_Gini(X,a,b):
    M_sample = 2000
    res=0
    N= len(X)
    
    for k in range(M_sample):
        theta = (0.5*np.pi)*np.random.beta(a, b, size=None)
        v_1=np.cos(theta)
        v_2=np.sin(theta)
        
        G=np.dot(X,(v_1,v_2))
        
        res=gini(G)*(1/(k+1))+res*((k)/(k+1))
        
    return res

def Sliced_beta_l2l1(X,a,b):
    M_sample = 2000
    res=0
    N= len(X)
    
    for k in range(M_sample):
        theta = (0.5*np.pi)*np.random.beta(a, b, size=None)
        v_1=np.cos(theta)
        v_2=np.sin(theta)
        
        G=np.dot(X,(v_1,v_2))
        
        res=l2l1(G)*(1/(k+1))+res*((k)/(k+1))
        
    return res

def Sliced_beta_Pietra(X,a,b):
    M_sample = 2000
    res=0
    N= len(X)
    
    for k in range(M_sample):
        theta = (0.5*np.pi)*np.random.beta(a, b, size=None)
        v_1=np.cos(theta)
        v_2=np.sin(theta)
        
        G=np.dot(X,(v_1,v_2))
        
        res=pietra(G)*(1/(k+1))+res*((k)/(k+1))
        
    return res

def Sliced_beta_Sinf(X,a,b):
    M_sample = 2000
    res=0
    N= len(X)
    
    for k in range(M_sample):
        theta = (0.5*np.pi)*np.random.beta(a, b, size=None)
        v_1=np.cos(theta)
        v_2=np.sin(theta)
        
        G=np.dot(X,(v_1,v_2))
        
        res=sinf(G)*(1/(k+1))+res*((k)/(k+1))
        # print(res)
        
    return res



#%%

# Reproducibility
rng = np.random.default_rng(seed=42)

# Number of samples
n = 1000


# -------------------------------------------------------
# 1.a Independent 2D Gamma distribution (positive orthant)
# -------------------------------------------------------
# X ~ Gamma(k1, theta1), Y ~ Gamma(k2, theta2)

k1, theta1 = 2.0, 1.0
k2, theta2 = 3.0, 0.5

gamma_samples = np.column_stack([
    rng.gamma(shape=k1, scale=theta1, size=n),
    rng.gamma(shape=k2, scale=theta2, size=n),
])


# -------------------------------------------------------
# 1.b Dependent 2D Gamma distribution
# -------------------------------------------------------
# X = exp(N(mu1, sigma1^2)), Y = exp(N(mu2, sigma2^2))

rho = 0.8

# Step 1: correlated normals
cov = [[1, rho], [rho, 1]]
z = rng.multivariate_normal([0, 0], cov, size=n)

# Step 2: transform to uniforms
u = norm.cdf(z)

# Step 3: transform to Gamma marginals
x = gamma.ppf(u[:, 0], a=2.0, scale=1.0)
y = gamma.ppf(u[:, 1], a=4.0, scale=0.5)

gamma_dep_samples = np.column_stack([x, y])


# -------------------------------------------------------
# 2.a Independent 2D Lognormal distribution
# -------------------------------------------------------
# X = exp(N(mu1, sigma1^2)), Y = exp(N(mu2, sigma2^2))

mu1, sigma1 = 0.0, 0.5
mu2, sigma2 = 0.5, 0.8

lognormal_samples = np.column_stack([
    rng.lognormal(mean=mu1, sigma=sigma1, size=n),
    rng.lognormal(mean=mu2, sigma=sigma2, size=n),
])


# -------------------------------------------------------
# 2.b Dependent 2D Lognormal distribution
# -------------------------------------------------------
# X = exp(N(mu1, sigma1^2)), Y = exp(N(mu2, sigma2^2))

mu = np.array([0.0, 0.5])
sigma1, sigma2 = 0.6, 1.0
rho = 0.7

Sigma = [
    [sigma1**2, rho * sigma1 * sigma2],
    [rho * sigma1 * sigma2, sigma2**2],
]

Z = rng.multivariate_normal(mu, Sigma, size=n)
X = np.exp(Z)   # dependent lognormals

log_dep_samples = X


# -------------------------------------------------------
# 3. Correlated 2D distribution using a Gaussian copula
#    with Gamma marginals
# -------------------------------------------------------
# Step 1: sample correlated normals
rho = 0.7
cov = [[1, rho],
       [rho, 1]]

z = rng.multivariate_normal(mean=[0, 0], cov=cov, size=n)

# Step 2: convert to uniform via Normal CDF
u = norm.cdf(z)

# Step 3: apply inverse Gamma CDF (PPF)
k1, theta1 = 2.0, 1.0
k2, theta2 = 5.0, 0.7

copula_gamma_samples = np.column_stack([
    gamma.ppf(u[:, 0], a=k1, scale=theta1),
    gamma.ppf(u[:, 1], a=k2, scale=theta2),
])


# -------------------------------------------------------
# Example output inspection
# -------------------------------------------------------
print("Gamma samples:\n", gamma_samples[:5])
print("\nLognormal samples:\n", lognormal_samples[:5])
print("\nCopula-Gamma samples:\n", copula_gamma_samples[:5])




#%%


# -------------------------------------------------------
#  NUMBER OF SAMPLE EXPERIMENTS
# -------------------------------------------------------



# N_sample =[50, 75, 100, 125, 150, 175, 200, 225, 250, 275, 300, 325, 350, 375, 400, 425, 450, 475, 500]
N_sample =[25, 50, 75, 100, 125, 150, 175, 200, 225, 250]
# N_sample =[250]

AVG_num = 200



# -------------------------------------------------------
#  GINI SLICED
# -------------------------------------------------------

SUG_indip_gamma = np.zeros((len(N_sample),AVG_num))
SBG_indip_gamma = np.zeros((len(N_sample),AVG_num))
SB26G_indip_gamma = np.zeros((len(N_sample),AVG_num))
SB62G_indip_gamma = np.zeros((len(N_sample),AVG_num))
SQG_indip_gamma = np.zeros((len(N_sample),AVG_num))
SDPG_indip_gamma = np.zeros((len(N_sample),AVG_num))

SUG_dip_gamma = np.zeros((len(N_sample),AVG_num))
SBG_dip_gamma = np.zeros((len(N_sample),AVG_num))
SB26G_dip_gamma = np.zeros((len(N_sample),AVG_num))
SB62G_dip_gamma = np.zeros((len(N_sample),AVG_num))
SQG_dip_gamma = np.zeros((len(N_sample),AVG_num))
SDPG_dip_gamma = np.zeros((len(N_sample),AVG_num))

SUG_indip_log = np.zeros((len(N_sample),AVG_num))
SBG_indip_log = np.zeros((len(N_sample),AVG_num))
SB26G_indip_log = np.zeros((len(N_sample),AVG_num))
SB62G_indip_log = np.zeros((len(N_sample),AVG_num))
SQG_indip_log = np.zeros((len(N_sample),AVG_num))
SDPG_indip_log = np.zeros((len(N_sample),AVG_num))

SUG_dip_log = np.zeros((len(N_sample),AVG_num))
SBG_dip_log = np.zeros((len(N_sample),AVG_num))
SB26G_dip_log = np.zeros((len(N_sample),AVG_num))
SB62G_dip_log = np.zeros((len(N_sample),AVG_num))
SQG_dip_log = np.zeros((len(N_sample),AVG_num))
SDPG_dip_log = np.zeros((len(N_sample),AVG_num))

SUG_dip_gauss = np.zeros((len(N_sample),AVG_num))
SBG_dip_gauss = np.zeros((len(N_sample),AVG_num))


# -------------------------------------------------------
#  PIETRA SLICED
# -------------------------------------------------------

SUP_indip_gamma = np.zeros((len(N_sample),AVG_num))
SBP_indip_gamma = np.zeros((len(N_sample),AVG_num))
SB26P_indip_gamma = np.zeros((len(N_sample),AVG_num))
SB62P_indip_gamma = np.zeros((len(N_sample),AVG_num))
SQP_indip_gamma = np.zeros((len(N_sample),AVG_num))
SDPP_indip_gamma = np.zeros((len(N_sample),AVG_num))

SUP_dip_gamma = np.zeros((len(N_sample),AVG_num))
SBP_dip_gamma = np.zeros((len(N_sample),AVG_num))
SB26P_dip_gamma = np.zeros((len(N_sample),AVG_num))
SB62P_dip_gamma = np.zeros((len(N_sample),AVG_num))
SQP_dip_gamma = np.zeros((len(N_sample),AVG_num))
SDPP_dip_gamma = np.zeros((len(N_sample),AVG_num))

SUP_indip_log = np.zeros((len(N_sample),AVG_num))
SBP_indip_log = np.zeros((len(N_sample),AVG_num))
SB26P_indip_log = np.zeros((len(N_sample),AVG_num))
SB62P_indip_log = np.zeros((len(N_sample),AVG_num))
SQP_indip_log = np.zeros((len(N_sample),AVG_num))
SDPP_indip_log = np.zeros((len(N_sample),AVG_num))

SUP_dip_log = np.zeros((len(N_sample),AVG_num))
SBP_dip_log = np.zeros((len(N_sample),AVG_num))
SB26P_dip_log = np.zeros((len(N_sample),AVG_num))
SB62P_dip_log = np.zeros((len(N_sample),AVG_num))
SQP_dip_log = np.zeros((len(N_sample),AVG_num))
SDPP_dip_log = np.zeros((len(N_sample),AVG_num))

SUG_dip_gauss = np.zeros((len(N_sample),AVG_num))
SBG_dip_gauss = np.zeros((len(N_sample),AVG_num))


# -------------------------------------------------------
#  Sinf SLICED
# -------------------------------------------------------

SUSinf_indip_gamma = np.zeros((len(N_sample),AVG_num))
SBSinf_indip_gamma = np.zeros((len(N_sample),AVG_num))
SB26Sinf_indip_gamma = np.zeros((len(N_sample),AVG_num))
SB62Sinf_indip_gamma = np.zeros((len(N_sample),AVG_num))
SQSinf_indip_gamma = np.zeros((len(N_sample),AVG_num))
SDPSinf_indip_gamma = np.zeros((len(N_sample),AVG_num))

SUSinf_dip_gamma = np.zeros((len(N_sample),AVG_num))
SBSinf_dip_gamma = np.zeros((len(N_sample),AVG_num))
SB26Sinf_dip_gamma = np.zeros((len(N_sample),AVG_num))
SB62Sinf_dip_gamma = np.zeros((len(N_sample),AVG_num))
SQSinf_dip_gamma = np.zeros((len(N_sample),AVG_num))
SDPSinf_dip_gamma = np.zeros((len(N_sample),AVG_num))

SUSinf_indip_log = np.zeros((len(N_sample),AVG_num))
SBSinf_indip_log = np.zeros((len(N_sample),AVG_num))
SB26Sinf_indip_log = np.zeros((len(N_sample),AVG_num))
SB62Sinf_indip_log = np.zeros((len(N_sample),AVG_num))
SQSinf_indip_log = np.zeros((len(N_sample),AVG_num))
SDPSinf_indip_log = np.zeros((len(N_sample),AVG_num))

SUSinf_dip_log = np.zeros((len(N_sample),AVG_num))
SBSinf_dip_log = np.zeros((len(N_sample),AVG_num))
SB26Sinf_dip_log = np.zeros((len(N_sample),AVG_num))
SB62Sinf_dip_log = np.zeros((len(N_sample),AVG_num))
SQSinf_dip_log = np.zeros((len(N_sample),AVG_num))
SDPSinf_dip_log = np.zeros((len(N_sample),AVG_num))

SUG_dip_gauss = np.zeros((len(N_sample),AVG_num))
SBG_dip_gauss = np.zeros((len(N_sample),AVG_num))


# -------------------------------------------------------
#  l2l1 SLICED
# -------------------------------------------------------

SUl2l1_indip_gamma = np.zeros((len(N_sample),AVG_num))
SBl2l1_indip_gamma = np.zeros((len(N_sample),AVG_num))
SB26l2l1_indip_gamma = np.zeros((len(N_sample),AVG_num))
SB62l2l1_indip_gamma = np.zeros((len(N_sample),AVG_num))
SQl2l1_indip_gamma = np.zeros((len(N_sample),AVG_num))
SDPl2l1_indip_gamma = np.zeros((len(N_sample),AVG_num))

SUl2l1_dip_gamma = np.zeros((len(N_sample),AVG_num))
SBl2l1_dip_gamma = np.zeros((len(N_sample),AVG_num))
SB26l2l1_dip_gamma = np.zeros((len(N_sample),AVG_num))
SB62l2l1_dip_gamma = np.zeros((len(N_sample),AVG_num))
SQl2l1_dip_gamma = np.zeros((len(N_sample),AVG_num))
SDPl2l1_dip_gamma = np.zeros((len(N_sample),AVG_num))

SUl2l1_indip_log = np.zeros((len(N_sample),AVG_num))
SBl2l1_indip_log = np.zeros((len(N_sample),AVG_num))
SB26l2l1_indip_log = np.zeros((len(N_sample),AVG_num))
SB62l2l1_indip_log = np.zeros((len(N_sample),AVG_num))
SQl2l1_indip_log = np.zeros((len(N_sample),AVG_num))
SDPl2l1_indip_log = np.zeros((len(N_sample),AVG_num))

SUl2l1_dip_log = np.zeros((len(N_sample),AVG_num))
SBl2l1_dip_log = np.zeros((len(N_sample),AVG_num))
SB26l2l1_dip_log = np.zeros((len(N_sample),AVG_num))
SB62l2l1_dip_log = np.zeros((len(N_sample),AVG_num))
SQl2l1_dip_log = np.zeros((len(N_sample),AVG_num))
SDPl2l1_dip_log = np.zeros((len(N_sample),AVG_num))

SUG_dip_gauss = np.zeros((len(N_sample),AVG_num))
SBG_dip_gauss = np.zeros((len(N_sample),AVG_num))



#%%

for j in range(len(N_sample)):
    
    n=N_sample[j]

    # -------------------------------------------------------
    # 1.a Independent 2D Gamma distribution (positive orthant)
    # -------------------------------------------------------
    # X ~ Gamma(k1, theta1), Y ~ Gamma(k2, theta2)

    k1, theta1 = 2.0, 1.0
    k2, theta2 = 3.0, 0.5

    # gamma_samples = np.column_stack([
    #     rng.gamma(shape=k1, scale=theta1, size=n),
    #     rng.gamma(shape=k2, scale=theta2, size=n),
    # ])

    for k in range(AVG_num):
        
        gamma_samples = np.column_stack([
            rng.gamma(shape=k1, scale=theta1, size=n),
            rng.gamma(shape=k2, scale=theta2, size=n),
        ])
        
        
    
        
        
        SUP_indip_gamma[j,k] = Sliced_unif_Pietra(gamma_samples)
        SBP_indip_gamma[j,k] = Sliced_beta_Pietra(gamma_samples,6,6)
        SB26P_indip_gamma[j,k] = Sliced_beta_Pietra(gamma_samples,2,6)
        SB62P_indip_gamma[j,k] = Sliced_beta_Pietra(gamma_samples,6,2)
        SQP_indip_gamma[j,k] = Sliced_quad_Pietra(gamma_samples)
        SDPP_indip_gamma[j,k] = Sliced_dp_Pietra(gamma_samples)
        
        
        
        
        
    
    # ------------------------------uuuu-------------------------
    # 1.b Dependent 2D Gamma distribution
    # -------------------------------------------------------
    # X = exp(N(mu1, sigma1^2)), Y = exp(N(mu2, sigma2^2))

    rho = 0.8

    # Step 1: correlated normals
    cov = [[1, rho], [rho, 1]]
   
    
    for k in range(AVG_num):
        
        z = rng.multivariate_normal([0, 0], cov, size=n)

        # Step 2: transform to uniforms
        u = norm.cdf(z)

        # Step 3: transform to Gamma marginals
        x = gamma.ppf(u[:, 0], a=2.0, scale=1.0)
        y = gamma.ppf(u[:, 1], a=4.0, scale=0.5)

        gamma_dep_samples = np.column_stack([x, y])
        
        
        
        
        
        SUP_dip_gamma[j,k] = Sliced_unif_Pietra(gamma_dep_samples)
        SBP_dip_gamma[j,k] = Sliced_beta_Pietra(gamma_dep_samples,6,6)
        SB26P_dip_gamma[j,k] = Sliced_beta_Pietra(gamma_dep_samples,2,6)
        SB62P_dip_gamma[j,k] = Sliced_beta_Pietra(gamma_dep_samples,6,2)
        SQP_dip_gamma[j,k] = Sliced_quad_Pietra(gamma_dep_samples)
        SDPP_dip_gamma[j,k] = Sliced_dp_Pietra(gamma_dep_samples)
        
        
        
        
        

    # -------------------------------------------------------
    # 2.a Independent 2D Lognormal distribution
    # -------------------------------------------------------
    # X = exp(N(mu1, sigma1^2)), Y = exp(N(mu2, sigma2^2))

    mu1, sigma1 = 0.0, 0.5
    mu2, sigma2 = 0.5, 0.8

    
    
    for k in range(AVG_num):
        
        lognormal_samples = np.column_stack([
            rng.lognormal(mean=mu1, sigma=sigma1, size=n),
            rng.lognormal(mean=mu2, sigma=sigma2, size=n),
        ])
        
        SUG_indip_log[j] = Sliced_unif_Gini(lognormal_samples)
        SBG_indip_log[j] = Sliced_beta_Gini(lognormal_samples,1,3)



    # -------------------------------------------------------
    # 2.b Dependent 2D Lognormal distribution
    # -------------------------------------------------------
    # X = exp(N(mu1, sigma1^2)), Y = exp(N(mu2, sigma2^2))

    mu = np.array([0.0, 0.5])
    sigma1, sigma2 = 0.6, 1.0
    rho = 0.7

    Sigma = [
        [sigma1**2, rho * sigma1 * sigma2],
        [rho * sigma1 * sigma2, sigma2**2],
    ]

    

   
    
    for k in range(AVG_num):
        
        Z = rng.multivariate_normal(mu, Sigma, size=n)
        X = np.exp(Z)   # dependent lognormals

        log_dep_samples = X
        
      
        
        
        
        SUP_dip_log[j,k] = Sliced_unif_Pietra(log_dep_samples)
        SBP_dip_log[j,k] = Sliced_beta_Pietra(log_dep_samples,6,6)
        SB26P_dip_log[j,k] = Sliced_beta_Pietra(log_dep_samples,2,6)
        SB62P_dip_log[j,k] = Sliced_beta_Pietra(log_dep_samples,6,2)
        SQP_dip_log[j,k] = Sliced_quad_Pietra(log_dep_samples)
        SDPP_dip_log[j,k] = Sliced_dp_Pietra(log_dep_samples)
        
        
        
        
        

#%%




# ============================================================
# INPUT DATA

#     GINI

# ============================================================


data1 = np.asarray(SUG_dip_log)
data2 = np.asarray(SBG_dip_log)
data3 = np.asarray(SB26G_dip_log)
data4 = np.asarray(SB62G_dip_log)
data5 = np.asarray(SQG_dip_log)
data6 = np.asarray(SDPG_dip_log)


# ============================================================
# X-AXIS VALUES
# ============================================================

# If i simply corresponds to 0, 1, 2, ..., use:
x =[25, 50, 75, 100, 125, 150, 175, 200, 225, 250]




# ============================================================
# COMPUTE MEAN AND 95% CONFIDENCE INTERVAL
# ============================================================

# Mean over the aleatory realizations (axis j)
mean1 = np.mean(data1, axis=1)
mean2 = np.mean(data2, axis=1)
mean3 = np.mean(data3, axis=1)
mean4 = np.mean(data4, axis=1)
mean5 = np.mean(data5, axis=1)
mean6 = np.mean(data6, axis=1)

# Number of realizations for each experiment
n = data1.shape[1]

# Standard deviation over j
std1 = np.std(data1, axis=1, ddof=1)
std2 = np.std(data2, axis=1, ddof=1)
std3 = np.std(data3, axis=1, ddof=1)
std4 = np.std(data4, axis=1, ddof=1)
std5 = np.std(data5, axis=1, ddof=1)
std6 = np.std(data6, axis=1, ddof=1)

# Standard error of the mean
sem1 = std1 / np.sqrt(n)
sem2 = std2 / np.sqrt(n)
sem3 = std3 / np.sqrt(n)
sem4 = std4 / np.sqrt(n)
sem5 = std5 / np.sqrt(n)
sem6 = std6 / np.sqrt(n)

# 95% confidence interval using Student's t distribution
confidence_level = 0.95
alpha = 1.0 - confidence_level

# t_critical = stats.t.ppf(1.0 - alpha / 2.0, df=n - 1)

z_critical = stats.norm.ppf(1.0 - alpha / 2.0)

margin1 = z_critical * sem1
margin2 = z_critical * sem2
margin3 = z_critical * sem3
margin4 = z_critical * sem4
margin5 = z_critical * sem5
margin6 = z_critical * sem6

lower1 = mean1 - margin1
upper1 = mean1 + margin1

GINI_lower1 = mean1 - margin1
GINI_upper1 = mean1 + margin1

lower2 = mean2 - margin2
upper2 = mean2 + margin2

GINI_lower2 = mean2 - margin2
GINI_upper2 = mean2 + margin2

lower3 = mean3 - margin3
upper3 = mean3 + margin3


GINI_lower3 = mean3 - margin3
GINI_upper3 = mean3 + margin3


lower4 = mean4 - margin4
upper4 = mean4 + margin4

lower5 = mean5 - margin5
upper5 = mean5 + margin5

lower6 = mean6 - margin6
upper6 = mean6 + margin6

GINI_lower4 = mean4 - margin4
GINI_upper4 = mean4 + margin4

GINI_lower5 = mean5 - margin5
GINI_upper5 = mean5 + margin5

GINI_lower6 = mean6 - margin6
GINI_upper6 = mean6 + margin6




GINI_range1 = GINI_upper1 - GINI_lower1 
GINI_range2 = GINI_upper2 - GINI_lower2 
GINI_range3 = GINI_upper3 - GINI_lower3 
GINI_range4 = GINI_upper4 - GINI_lower4 
GINI_range5 = GINI_upper5 - GINI_lower5 
GINI_range6 = GINI_upper6 - GINI_lower6 


print('------------SINF-------------------')


print(np.mean(GINI_range1),np.max(GINI_range1))
print(np.mean(GINI_range2),np.max(GINI_range2))
print(np.mean(GINI_range3),np.max(GINI_range3))
print(np.mean(GINI_range4),np.max(GINI_range4))
print(np.mean(GINI_range5),np.max(GINI_range5))
print(np.mean(GINI_range6),np.max(GINI_range6))


print('-------------------------------------')


# ============================================================
# PLOT
# ============================================================

fig, ax = plt.subplots(figsize=(7, 5))

# Mean
ax.plot(
    x,
    mean1,
    marker='o',
    linewidth=1.8,
    markersize=5,
    label='Uniform'
)

# Confidence interval
ax.fill_between(
    x,
    lower1,
    upper1,
    alpha=0.25,
    # label='95% confidence interval'
)

# Mean
ax.plot(
    x,
    mean2,
    marker='o',
    linewidth=1.8,
    markersize=5,
    label='Beta(6,6)'
)

# Confidence interval
ax.fill_between(
    x,
    lower2,
    upper2,
    alpha=0.25,
    # label='95% confidence interval'
)

# Mean
ax.plot(
    x,
    mean3,
    marker='o',
    linewidth=1.8,
    markersize=5,
    label='Beta(2,6)'
)

# Confidence interval
ax.fill_between(
    x,
    lower3,
    upper3,
    alpha=0.25,
    # label='95% confidence interval'
)

# Mean
ax.plot(
    x,
    mean4,
    marker='o',
    linewidth=1.8,
    markersize=5,
    label='Beta(6,2)'
)

# Confidence interval
ax.fill_between(
    x,
    lower4,
    upper4,
    alpha=0.25,
    # label='95% confidence interval'
)

# Mean
ax.plot(
    x,
    mean5,
    marker='o',
    linewidth=1.8,
    markersize=5,
    label='Quadratic'
)

# Confidence interval
ax.fill_between(
    x,
    lower5,
    upper5,
    alpha=0.25,
    # label='95% confidence interval'
)

# Mean
ax.plot(
    x,
    mean6,
    marker='o',
    linewidth=1.8,
    markersize=5,
    label='Double Peak'
)

# Confidence interval
ax.fill_between(
    x,
    lower6,
    upper6,
    alpha=0.25,
    # label='95% confidence interval'
)


# ============================================================
# AXIS LABELS AND STYLE
# ============================================================

ax.set_xlabel(r'Sample size', fontsize=12)
ax.set_ylabel(r'Sliced Gini', fontsize=12)

ax.grid(True, alpha=0.3)
ax.legend()

fig.tight_layout()


# ============================================================
# SAVE AS PDF
# ============================================================

fig.savefig(
    'GINI_slsp_log_dip.pdf',
    format='pdf',
    bbox_inches='tight'
)

plt.show()


# ============================================================
# INPUT DATA
# ============================================================



data1 = np.asarray(SUG_dip_gamma)
data2 = np.asarray(SBG_dip_gamma)
data3 = np.asarray(SB26G_dip_gamma)
data4 = np.asarray(SB62G_dip_gamma)
data5 = np.asarray(SQG_dip_gamma)
data6 = np.asarray(SDPG_dip_gamma)


# ============================================================
# X-AXIS VALUES
# ============================================================

# If i simply corresponds to 0, 1, 2, ..., use:
x =[25, 50, 75, 100, 125, 150, 175, 200, 225, 250]




# ============================================================
# COMPUTE MEAN AND 95% CONFIDENCE INTERVAL
# ============================================================

# Mean over the aleatory realizations (axis j)
mean1 = np.mean(data1, axis=1)
mean2 = np.mean(data2, axis=1)
mean3 = np.mean(data3, axis=1)
mean4 = np.mean(data4, axis=1)
mean5 = np.mean(data5, axis=1)
mean6 = np.mean(data6, axis=1)

# Number of realizations for each experiment
n = data1.shape[1]

# Standard deviation over j
std1 = np.std(data1, axis=1, ddof=1)
std2 = np.std(data2, axis=1, ddof=1)
std3 = np.std(data3, axis=1, ddof=1)
std4 = np.std(data4, axis=1, ddof=1)
std5 = np.std(data5, axis=1, ddof=1)
std6 = np.std(data6, axis=1, ddof=1)

# Standard error of the mean
sem1 = std1 / np.sqrt(n)
sem2 = std2 / np.sqrt(n)
sem3 = std3 / np.sqrt(n)
sem4 = std4 / np.sqrt(n)
sem5 = std5 / np.sqrt(n)
sem6 = std6 / np.sqrt(n)

# 95% confidence interval using Student's t distribution
confidence_level = 0.95
alpha = 1.0 - confidence_level

# t_critical = stats.t.ppf(1.0 - alpha / 2.0, df=n - 1)
z_critical = stats.norm.ppf(1.0 - alpha / 2.0)

margin1 = z_critical * sem1
margin2 = z_critical * sem2
margin3 = z_critical * sem3
margin4 = z_critical * sem4
margin5 = z_critical * sem5
margin6 = z_critical * sem6

lower1 = mean1 - margin1
upper1 = mean1 + margin1

lower2 = mean2 - margin2
upper2 = mean2 + margin2

lower3 = mean3 - margin3
upper3 = mean3 + margin3

lower4 = mean4 - margin4
upper4 = mean4 + margin4

lower5 = mean5 - margin5
upper5 = mean5 + margin5

lower6 = mean6 - margin6
upper6 = mean6 + margin6



PIETRA_lower1 = mean1 - margin1
PIETRA_upper1 = mean1 + margin1

PIETRA_lower2 = mean2 - margin2
PIETRA_upper2 = mean2 + margin2

PIETRA_lower3 = mean3 - margin3
PIETRA_upper3 = mean3 + margin3

PIETRA_lower4 = mean4 - margin4
PIETRA_upper4 = mean4 + margin4

PIETRA_lower5 = mean5 - margin5
PIETRA_upper5 = mean5 + margin5

lower6 = mean6 - margin6
upper6 = mean6 + margin6


# ============================================================
# PLOT
# ============================================================

fig, ax = plt.subplots(figsize=(7, 5))

# Mean
ax.plot(
    x,
    mean1,
    marker='o',
    linewidth=1.8,
    markersize=5,
    label='Uniform'
)

# Confidence interval
ax.fill_between(
    x,
    lower1,
    upper1,
    alpha=0.25,
    # label='95% confidence interval'
)

# Mean
ax.plot(
    x,
    mean2,
    marker='o',
    linewidth=1.8,
    markersize=5,
    label='Beta(6,6)'
)

# Confidence interval
ax.fill_between(
    x,
    lower2,
    upper2,
    alpha=0.25,
    # label='95% confidence interval'
)

# Mean
ax.plot(
    x,
    mean3,
    marker='o',
    linewidth=1.8,
    markersize=5,
    label='Beta(2,6)'
)

# Confidence interval
ax.fill_between(
    x,
    lower3,
    upper3,
    alpha=0.25,
    # label='95% confidence interval'
)

# Mean
ax.plot(
    x,
    mean4,
    marker='o',
    linewidth=1.8,
    markersize=5,
    label='Beta(6,2)'
)

# Confidence interval
ax.fill_between(
    x,
    lower4,
    upper4,
    alpha=0.25,
    # label='95% confidence interval'
)

# Mean
ax.plot(
    x,
    mean5,
    marker='o',
    linewidth=1.8,
    markersize=5,
    label='Quadratic'
)

# Confidence interval
ax.fill_between(
    x,
    lower5,
    upper5,
    alpha=0.25,
    # label='95% confidence interval'
)

# Mean
ax.plot(
    x,
    mean6,
    marker='o',
    linewidth=1.8,
    markersize=5,
    label='Double Peak'
)

# Confidence interval
ax.fill_between(
    x,
    lower6,
    upper6,
    alpha=0.25,
    # label='95% confidence interval'
)


# ============================================================
# AXIS LABELS AND STYLE
# ============================================================

ax.set_xlabel(r'Sample size', fontsize=12)
ax.set_ylabel(r'Sliced Gini', fontsize=12)

ax.grid(True, alpha=0.3)
ax.legend()

fig.tight_layout()


# ============================================================
# SAVE AS PDF
# ============================================================

fig.savefig(
    'GINI_slsp_gamma_dip.pdf',
    format='pdf',
    bbox_inches='tight'
)

plt.show()



# ============================================================
# INPUT DATA
# ============================================================



data1 = np.asarray(SUG_indip_gamma)
data2 = np.asarray(SBG_indip_gamma)
data3 = np.asarray(SB26G_indip_gamma)
data4 = np.asarray(SB62G_indip_gamma)
data5 = np.asarray(SQG_indip_gamma)
data6 = np.asarray(SDPG_indip_gamma)


# ============================================================
# X-AXIS VALUES
# ============================================================

x =[25, 50, 75, 100, 125, 150, 175, 200, 225, 250]



# ============================================================
# COMPUTE MEAN AND 95% CONFIDENCE INTERVAL
# ============================================================

# Mean over the aleatory realizations (axis j)
mean1 = np.mean(data1, axis=1)
mean2 = np.mean(data2, axis=1)
mean3 = np.mean(data3, axis=1)
mean4 = np.mean(data4, axis=1)
mean5 = np.mean(data5, axis=1)
mean6 = np.mean(data6, axis=1)

# Number of realizations for each experiment
n = data1.shape[1]

# Standard deviation over j
std1 = np.std(data1, axis=1, ddof=1)
std2 = np.std(data2, axis=1, ddof=1)
std3 = np.std(data3, axis=1, ddof=1)
std4 = np.std(data4, axis=1, ddof=1)
std5 = np.std(data5, axis=1, ddof=1)
std6 = np.std(data6, axis=1, ddof=1)

# Standard error of the mean
sem1 = std1 / np.sqrt(n)
sem2 = std2 / np.sqrt(n)
sem3 = std3 / np.sqrt(n)
sem4 = std4 / np.sqrt(n)
sem5 = std5 / np.sqrt(n)
sem6 = std6 / np.sqrt(n)

# 95% confidence interval using Student's t distribution
confidence_level = 0.95
alpha = 1.0 - confidence_level

# t_critical = stats.t.ppf(1.0 - alpha / 2.0, df=n - 1)
z_critical = stats.norm.ppf(1.0 - alpha / 2.0)

margin1 = z_critical * sem1
margin2 = z_critical * sem2
margin3 = z_critical * sem3
margin4 = z_critical * sem4
margin5 = z_critical * sem5
margin6 = z_critical * sem6

lower1 = mean1 - margin1
upper1 = mean1 + margin1

lower2 = mean2 - margin2
upper2 = mean2 + margin2

lower3 = mean3 - margin3
upper3 = mean3 + margin3

lower4 = mean4 - margin4
upper4 = mean4 + margin4

lower5 = mean5 - margin5
upper5 = mean5 + margin5

lower6 = mean6 - margin6
upper6 = mean6 + margin6


# ============================================================
# PLOT
# ============================================================

fig, ax = plt.subplots(figsize=(7, 5))

# Mean
ax.plot(
    x,
    mean1,
    marker='o',
    linewidth=1.8,
    markersize=5,
    label='Uniform'
)

# Confidence interval
ax.fill_between(
    x,
    lower1,
    upper1,
    alpha=0.25,
    # label='95% confidence interval'
)

# Mean
ax.plot(
    x,
    mean2,
    marker='o',
    linewidth=1.8,
    markersize=5,
    label='Beta(6,6)'
)

# Confidence interval
ax.fill_between(
    x,
    lower2,
    upper2,
    alpha=0.25,
    # label='95% confidence interval'
)

# Mean
ax.plot(
    x,
    mean3,
    marker='o',
    linewidth=1.8,
    markersize=5,
    label='Beta(2,6)'
)

# Confidence interval
ax.fill_between(
    x,
    lower3,
    upper3,
    alpha=0.25,
    # label='95% confidence interval'
)

# Mean
ax.plot(
    x,
    mean4,
    marker='o',
    linewidth=1.8,
    markersize=5,
    label='Beta(6,2)'
)

# Confidence interval
ax.fill_between(
    x,
    lower4,
    upper4,
    alpha=0.25,
    # label='95% confidence interval'
)

# Mean
ax.plot(
    x,
    mean5,
    marker='o',
    linewidth=1.8,
    markersize=5,
    label='Quadratic'
)

# Confidence interval
ax.fill_between(
    x,
    lower5,
    upper5,
    alpha=0.25,
    # label='95% confidence interval'
)

# Mean
ax.plot(
    x,
    mean6,
    marker='o',
    linewidth=1.8,
    markersize=5,
    label='Double Peak'
)

# Confidence interval
ax.fill_between(
    x,
    lower6,
    upper6,
    alpha=0.25,
    # label='95% confidence interval'
)


# ============================================================
# AXIS LABELS AND STYLE
# ============================================================

ax.set_xlabel(r'Sample size', fontsize=12)
ax.set_ylabel(r'Sliced Gini', fontsize=12)

ax.grid(True, alpha=0.3)
ax.legend()

fig.tight_layout()


# ============================================================
# SAVE AS PDF
# ============================================================

fig.savefig(
    'GINI_slsp_gamma_indip.pdf',
    format='pdf',
    bbox_inches='tight'
)

plt.show()

#%%


# ============================================================
# INPUT DATA

#     PIETRA

# ============================================================



data1 = np.asarray(SUP_dip_log)
data2 = np.asarray(SBP_dip_log)
data3 = np.asarray(SB26P_dip_log)
data4 = np.asarray(SB62P_dip_log)
data5 = np.asarray(SQP_dip_log)
data6 = np.asarray(SDPP_dip_log)


# ============================================================
# X-AXIS VALUES
# ============================================================

# 
x =[25, 50, 75, 100, 125, 150, 175, 200, 225, 250]




# ============================================================
# COMPUTE MEAN AND 95% CONFIDENCE INTERVAL
# ============================================================


mean1 = np.mean(data1, axis=1)
mean2 = np.mean(data2, axis=1)
mean3 = np.mean(data3, axis=1)
mean4 = np.mean(data4, axis=1)
mean5 = np.mean(data5, axis=1)
mean6 = np.mean(data6, axis=1)

# Number of realizations for each experiment
n = data1.shape[1]

# Standard deviation over j
std1 = np.std(data1, axis=1, ddof=1)
std2 = np.std(data2, axis=1, ddof=1)
std3 = np.std(data3, axis=1, ddof=1)
std4 = np.std(data4, axis=1, ddof=1)
std5 = np.std(data5, axis=1, ddof=1)
std6 = np.std(data6, axis=1, ddof=1)

# Standard error of the mean
sem1 = std1 / np.sqrt(n)
sem2 = std2 / np.sqrt(n)
sem3 = std3 / np.sqrt(n)
sem4 = std4 / np.sqrt(n)
sem5 = std5 / np.sqrt(n)
sem6 = std6 / np.sqrt(n)

# 95% confidence interval using Student's t distribution
confidence_level = 0.95
alpha = 1.0 - confidence_level

# t_critical = stats.t.ppf(1.0 - alpha / 2.0, df=n - 1)
z_critical = stats.norm.ppf(1.0 - alpha / 2.0)

margin1 = z_critical * sem1
margin2 = z_critical * sem2
margin3 = z_critical * sem3
margin4 = z_critical * sem4
margin5 = z_critical * sem5
margin6 = z_critical * sem6

lower1 = mean1 - margin1
upper1 = mean1 + margin1

lower2 = mean2 - margin2
upper2 = mean2 + margin2

lower3 = mean3 - margin3
upper3 = mean3 + margin3

lower4 = mean4 - margin4
upper4 = mean4 + margin4

lower5 = mean5 - margin5
upper5 = mean5 + margin5

lower6 = mean6 - margin6
upper6 = mean6 + margin6


print('-------------PIETRA------------------')

PIETRA_lower1 = mean1 - margin1
PIETRA_upper1 = mean1 + margin1

PIETRA_range1 = PIETRA_upper1 - PIETRA_lower1 

print(np.mean(PIETRA_range1),np.max(PIETRA_range1))

PIETRA_lower2 = mean2 - margin2
PIETRA_upper2 = mean2 + margin2

PIETRA_range2 = PIETRA_upper2 - PIETRA_lower2

print(np.mean(PIETRA_range2),np.max(PIETRA_range2))

PIETRA_lower3 = mean3 - margin3
PIETRA_upper3 = mean3 + margin3

PIETRA_range3 = PIETRA_upper3 - PIETRA_lower3

print(np.mean(PIETRA_range3),np.max(PIETRA_range3))

PIETRA_lower4 = mean4 - margin4
PIETRA_upper4 = mean4 + margin4

PIETRA_range4 = PIETRA_upper4 - PIETRA_lower4

print(np.mean(PIETRA_range4),np.max(PIETRA_range4))

PIETRA_lower5 = mean5 - margin5
PIETRA_upper5 = mean5 + margin5

PIETRA_range5 = PIETRA_upper5 - PIETRA_lower5

print(np.mean(PIETRA_range5),np.max(PIETRA_range5))

PIETRA_lower6 = mean6 - margin6
PIETRA_upper6 = mean6 + margin6

PIETRA_range6 = PIETRA_upper6 - PIETRA_lower6

print(np.mean(PIETRA_range6),np.max(PIETRA_range6))

print('-------------------------------------')


# ============================================================
# PLOT
# ============================================================

fig, ax = plt.subplots(figsize=(7, 5))

# Mean
ax.plot(
    x,
    mean1,
    marker='o',
    linewidth=1.8,
    markersize=5,
    label='Uniform'
)

# Confidence interval
ax.fill_between(
    x,
    lower1,
    upper1,
    alpha=0.25,
    # label='95% confidence interval'
)

# Mean
ax.plot(
    x,
    mean2,
    marker='o',
    linewidth=1.8,
    markersize=5,
    label='Beta(6,6)'
)

# Confidence interval
ax.fill_between(
    x,
    lower2,
    upper2,
    alpha=0.25,
    # label='95% confidence interval'
)

# Mean
ax.plot(
    x,
    mean3,
    marker='o',
    linewidth=1.8,
    markersize=5,
    label='Beta(2,6)'
)

# Confidence interval
ax.fill_between(
    x,
    lower3,
    upper3,
    alpha=0.25,
    # label='95% confidence interval'
)

# Mean
ax.plot(
    x,
    mean4,
    marker='o',
    linewidth=1.8,
    markersize=5,
    label='Beta(6,2)'
)

# Confidence interval
ax.fill_between(
    x,
    lower4,
    upper4,
    alpha=0.25,
    # label='95% confidence interval'
)

# Mean
ax.plot(
    x,
    mean5,
    marker='o',
    linewidth=1.8,
    markersize=5,
    label='Quadratic'
)

# Confidence interval
ax.fill_between(
    x,
    lower5,
    upper5,
    alpha=0.25,
    # label='95% confidence interval'
)

# Mean
ax.plot(
    x,
    mean6,
    marker='o',
    linewidth=1.8,
    markersize=5,
    label='Double Peak'
)

# Confidence interval
ax.fill_between(
    x,
    lower6,
    upper6,
    alpha=0.25,
    # label='95% confidence interval'
)


# ============================================================
# AXIS LABELS AND STYLE
# ============================================================

ax.set_xlabel(r'Sample size', fontsize=12)
ax.set_ylabel(r'Sliced Pietra', fontsize=12)

ax.grid(True, alpha=0.3)
ax.legend()

fig.tight_layout()


# ============================================================
# SAVE AS PDF
# ============================================================

fig.savefig(
    '1PIETRA_slsp_log_dip.pdf',
    format='pdf',
    bbox_inches='tight'
)

plt.show()


# ============================================================
# INPUT DATA
# ============================================================

#

data1 = np.asarray(SUP_dip_gamma)
data2 = np.asarray(SBP_dip_gamma)
data3 = np.asarray(SB26P_dip_gamma)
data4 = np.asarray(SB62P_dip_gamma)
data5 = np.asarray(SQP_dip_gamma)
data6 = np.asarray(SDPP_dip_gamma)


# ============================================================
# X-AXIS VALUES
# ============================================================

# 
x =[25, 50, 75, 100, 125, 150, 175, 200, 225, 250]

# Alternatively, if each i corresponds to specific experimental
# values, replace the previous line with something like:
#
# x = np.array([0.1, 0.2, 0.5, 1.0, 2.0])

# plt.plot(N_sample, SUG_dip_log, label="Uniform")
# plt.plot(N_sample, SBG_dip_log, label="Beta(6,6)")
# plt.plot(N_sample, SB26G_dip_log, label="Beta(2,6)")
# plt.plot(N_sample, SB62G_dip_log, label="Beta(6,2)")
# plt.plot(N_sample, SQG_dip_log, label="Quadratic")
# plt.plot(N_sample, SDPG_dip_log, label="Double Peak")


# ============================================================
# COMPUTE MEAN AND 95% CONFIDENCE INTERVAL
# ============================================================

# Mean over the aleatory realizations (axis j)
mean1 = np.mean(data1, axis=1)
mean2 = np.mean(data2, axis=1)
mean3 = np.mean(data3, axis=1)
mean4 = np.mean(data4, axis=1)
mean5 = np.mean(data5, axis=1)
mean6 = np.mean(data6, axis=1)

# Number of realizations for each experiment
n = data1.shape[1]

# Standard deviation over j
std1 = np.std(data1, axis=1, ddof=1)
std2 = np.std(data2, axis=1, ddof=1)
std3 = np.std(data3, axis=1, ddof=1)
std4 = np.std(data4, axis=1, ddof=1)
std5 = np.std(data5, axis=1, ddof=1)
std6 = np.std(data6, axis=1, ddof=1)

# Standard error of the mean
sem1 = std1 / np.sqrt(n)
sem2 = std2 / np.sqrt(n)
sem3 = std3 / np.sqrt(n)
sem4 = std4 / np.sqrt(n)
sem5 = std5 / np.sqrt(n)
sem6 = std6 / np.sqrt(n)

# 95% confidence interval using Student's t distribution
confidence_level = 0.95
alpha = 1.0 - confidence_level

# t_critical = stats.t.ppf(1.0 - alpha / 2.0, df=n - 1)

z_critical = stats.norm.ppf(1.0 - alpha / 2.0)


margin1 = z_critical * sem1
margin2 = z_critical * sem2
margin3 = z_critical * sem3
margin4 = z_critical * sem4
margin5 = z_critical * sem5
margin6 = z_critical * sem6

lower1 = mean1 - margin1
upper1 = mean1 + margin1

lower2 = mean2 - margin2
upper2 = mean2 + margin2

lower3 = mean3 - margin3
upper3 = mean3 + margin3

lower4 = mean4 - margin4
upper4 = mean4 + margin4

lower5 = mean5 - margin5
upper5 = mean5 + margin5

lower6 = mean6 - margin6
upper6 = mean6 + margin6


# ============================================================
# PLOT
# ============================================================

fig, ax = plt.subplots(figsize=(7, 5))

# Mean
ax.plot(
    x,
    mean1,
    marker='o',
    linewidth=1.8,
    markersize=5,
    label='Uniform'
)

# Confidence interval
ax.fill_between(
    x,
    lower1,
    upper1,
    alpha=0.25,
    # label='95% confidence interval'
)

# Mean
ax.plot(
    x,
    mean2,
    marker='o',
    linewidth=1.8,
    markersize=5,
    label='Beta(6,6)'
)

# Confidence interval
ax.fill_between(
    x,
    lower2,
    upper2,
    alpha=0.25,
    # label='95% confidence interval'
)

# Mean
ax.plot(
    x,
    mean3,
    marker='o',
    linewidth=1.8,
    markersize=5,
    label='Beta(2,6)'
)

# Confidence interval
ax.fill_between(
    x,
    lower3,
    upper3,
    alpha=0.25,
    # label='95% confidence interval'
)

# Mean
ax.plot(
    x,
    mean4,
    marker='o',
    linewidth=1.8,
    markersize=5,
    label='Beta(6,2)'
)

# Confidence interval
ax.fill_between(
    x,
    lower4,
    upper4,
    alpha=0.25,
    # label='95% confidence interval'
)

# Mean
ax.plot(
    x,
    mean5,
    marker='o',
    linewidth=1.8,
    markersize=5,
    label='Quadratic'
)

# Confidence interval
ax.fill_between(
    x,
    lower5,
    upper5,
    alpha=0.25,
    # label='95% confidence interval'
)

# Mean
ax.plot(
    x,
    mean6,
    marker='o',
    linewidth=1.8,
    markersize=5,
    label='Double Peak'
)

# Confidence interval
ax.fill_between(
    x,
    lower6,
    upper6,
    alpha=0.25,
    # label='95% confidence interval'
)


# ============================================================
# AXIS LABELS AND STYLE
# ============================================================

ax.set_xlabel(r'Sample size', fontsize=12)
ax.set_ylabel(r'Sliced Pietra', fontsize=12)

ax.grid(True, alpha=0.3)
ax.legend()

fig.tight_layout()


# ============================================================
# SAVE AS PDF
# ============================================================

fig.savefig(
    '1PIETRA_slsp_gamma_dip.pdf',
    format='pdf',
    bbox_inches='tight'
)

plt.show()



# ============================================================
# INPUT DATA
# ============================================================

# Your matrix:
# SUl2l1_indip_gamma[i, j]
#
# It should already be available in your Python environment.
#
# Example:
# SUl2l1_indip_gamma = np.array(...)

data1 = np.asarray(SUP_indip_gamma)
data2 = np.asarray(SBP_indip_gamma)
data3 = np.asarray(SB26P_indip_gamma)
data4 = np.asarray(SB62P_indip_gamma)
data5 = np.asarray(SQP_indip_gamma)
data6 = np.asarray(SDPP_indip_gamma)


# ============================================================
# X-AXIS VALUES
# ============================================================

# If i simply corresponds to 0, 1, 2, ..., use:
x =[25, 50, 75, 100, 125, 150, 175, 200, 225, 250]

# Alternatively, if each i corresponds to specific experimental
# values, replace the previous line with something like:
#
# x = np.array([0.1, 0.2, 0.5, 1.0, 2.0])

# plt.plot(N_sample, SUG_dip_log, label="Uniform")
# plt.plot(N_sample, SBG_dip_log, label="Beta(6,6)")
# plt.plot(N_sample, SB26G_dip_log, label="Beta(2,6)")
# plt.plot(N_sample, SB62G_dip_log, label="Beta(6,2)")
# plt.plot(N_sample, SQG_dip_log, label="Quadratic")
# plt.plot(N_sample, SDPG_dip_log, label="Double Peak")


# ============================================================
# COMPUTE MEAN AND 95% CONFIDENCE INTERVAL
# ============================================================

# Mean over the aleatory realizations (axis j)
mean1 = np.mean(data1, axis=1)
mean2 = np.mean(data2, axis=1)
mean3 = np.mean(data3, axis=1)
mean4 = np.mean(data4, axis=1)
mean5 = np.mean(data5, axis=1)
mean6 = np.mean(data6, axis=1)

# Number of realizations for each experiment
n = data1.shape[1]

# Standard deviation over j
std1 = np.std(data1, axis=1, ddof=1)
std2 = np.std(data2, axis=1, ddof=1)
std3 = np.std(data3, axis=1, ddof=1)
std4 = np.std(data4, axis=1, ddof=1)
std5 = np.std(data5, axis=1, ddof=1)
std6 = np.std(data6, axis=1, ddof=1)

# Standard error of the mean
sem1 = std1 / np.sqrt(n)
sem2 = std2 / np.sqrt(n)
sem3 = std3 / np.sqrt(n)
sem4 = std4 / np.sqrt(n)
sem5 = std5 / np.sqrt(n)
sem6 = std6 / np.sqrt(n)

# 95% confidence interval using Student's t distribution
confidence_level = 0.95
alpha = 1.0 - confidence_level

# t_critical = stats.t.ppf(1.0 - alpha / 2.0, df=n - 1)
z_critical = stats.norm.ppf(1.0 - alpha / 2.0)

margin1 = z_critical * sem1
margin2 = z_critical * sem2
margin3 = z_critical * sem3
margin4 = z_critical * sem4
margin5 = z_critical * sem5
margin6 = z_critical * sem6

lower1 = mean1 - margin1
upper1 = mean1 + margin1

lower2 = mean2 - margin2
upper2 = mean2 + margin2

lower3 = mean3 - margin3
upper3 = mean3 + margin3

lower4 = mean4 - margin4
upper4 = mean4 + margin4

lower5 = mean5 - margin5
upper5 = mean5 + margin5

lower6 = mean6 - margin6
upper6 = mean6 + margin6


# ============================================================
# PLOT
# ============================================================

fig, ax = plt.subplots(figsize=(7, 5))

# Mean
ax.plot(
    x,
    mean1,
    marker='o',
    linewidth=1.8,
    markersize=5,
    label='Uniform'
)

# Confidence interval
ax.fill_between(
    x,
    lower1,
    upper1,
    alpha=0.25,
    # label='95% confidence interval'
)

# Mean
ax.plot(
    x,
    mean2,
    marker='o',
    linewidth=1.8,
    markersize=5,
    label='Beta(6,6)'
)

# Confidence interval
ax.fill_between(
    x,
    lower2,
    upper2,
    alpha=0.25,
    # label='95% confidence interval'
)

# Mean
ax.plot(
    x,
    mean3,
    marker='o',
    linewidth=1.8,
    markersize=5,
    label='Beta(2,6)'
)

# Confidence interval
ax.fill_between(
    x,
    lower3,
    upper3,
    alpha=0.25,
    # label='95% confidence interval'
)

# Mean
ax.plot(
    x,
    mean4,
    marker='o',
    linewidth=1.8,
    markersize=5,
    label='Beta(6,2)'
)

# Confidence interval
ax.fill_between(
    x,
    lower4,
    upper4,
    alpha=0.25,
    # label='95% confidence interval'
)

# Mean
ax.plot(
    x,
    mean5,
    marker='o',
    linewidth=1.8,
    markersize=5,
    label='Quadratic'
)

# Confidence interval
ax.fill_between(
    x,
    lower5,
    upper5,
    alpha=0.25,
    # label='95% confidence interval'
)

# Mean
ax.plot(
    x,
    mean6,
    marker='o',
    linewidth=1.8,
    markersize=5,
    label='Double Peak'
)

# Confidence interval
ax.fill_between(
    x,
    lower6,
    upper6,
    alpha=0.25,
    # label='95% confidence interval'
)


# ============================================================
# AXIS LABELS AND STYLE
# ============================================================

ax.set_xlabel(r'Sample size', fontsize=12)
ax.set_ylabel(r'Sliced Pietra', fontsize=12)

ax.grid(True, alpha=0.3)
ax.legend()

fig.tight_layout()


# ============================================================
# SAVE AS PDF
# ============================================================

fig.savefig(
    '1PIETRA_slsp_gamma_indip.pdf',
    format='pdf',
    bbox_inches='tight'
)

plt.show()

print("Current working directory:", os.getcwd())
print("Full file path:", os.path.abspath('1PIETRA_slsp_log_dip.pdf'))
print("File exists:", os.path.exists('1PIETRA_slsp_log_dip.pdf'))


#%%


# ============================================================
# INPUT DATA

#     Sinf

# ============================================================

# Your matrix:
# SUl2l1_indip_gamma[i, j]
#
# It should already be available in your Python environment.
#
# Example:
# SUl2l1_indip_gamma = np.array(...)

data1 = np.asarray(SUSinf_dip_log)
data2 = np.asarray(SBSinf_dip_log)
data3 = np.asarray(SB26Sinf_dip_log)
data4 = np.asarray(SB62Sinf_dip_log)
data5 = np.asarray(SQSinf_dip_log)
data6 = np.asarray(SDPSinf_dip_log)


# ============================================================
# X-AXIS VALUES
# ============================================================

# If i simply corresponds to 0, 1, 2, ..., use:
x =[25, 50, 75, 100, 125, 150, 175, 200, 225, 250]

# 


# ============================================================
# COMPUTE MEAN AND 95% CONFIDENCE INTERVAL
# ============================================================

# Mean over the aleatory realizations (axis j)
mean1 = np.mean(data1, axis=1)
mean2 = np.mean(data2, axis=1)
mean3 = np.mean(data3, axis=1)
mean4 = np.mean(data4, axis=1)
mean5 = np.mean(data5, axis=1)
mean6 = np.mean(data6, axis=1)

# Number of realizations for each experiment
n = data1.shape[1]

# Standard deviation over j
std1 = np.std(data1, axis=1, ddof=1)
std2 = np.std(data2, axis=1, ddof=1)
std3 = np.std(data3, axis=1, ddof=1)
std4 = np.std(data4, axis=1, ddof=1)
std5 = np.std(data5, axis=1, ddof=1)
std6 = np.std(data6, axis=1, ddof=1)

# Standard error of the mean
sem1 = std1 / np.sqrt(n)
sem2 = std2 / np.sqrt(n)
sem3 = std3 / np.sqrt(n)
sem4 = std4 / np.sqrt(n)
sem5 = std5 / np.sqrt(n)
sem6 = std6 / np.sqrt(n)

# 95% confidence interval using Student's t distribution
confidence_level = 0.95
alpha = 1.0 - confidence_level

# t_critical = stats.t.ppf(1.0 - alpha / 2.0, df=n - 1)

z_critical = stats.norm.ppf(1.0 - alpha / 2.0)


margin1 = z_critical * sem1
margin2 = z_critical * sem2
margin3 = z_critical * sem3
margin4 = z_critical * sem4
margin5 = z_critical * sem5
margin6 = z_critical * sem6

lower1 = mean1 - margin1
upper1 = mean1 + margin1

lower2 = mean2 - margin2
upper2 = mean2 + margin2

lower3 = mean3 - margin3
upper3 = mean3 + margin3

lower4 = mean4 - margin4
upper4 = mean4 + margin4

lower5 = mean5 - margin5
upper5 = mean5 + margin5

lower6 = mean6 - margin6
upper6 = mean6 + margin6


SINF_lower1 = mean1 - margin1
SINF_upper1 = mean1 + margin1

SINF_range1 = SINF_lower1 -SINF_upper1

SINF_lower2 = mean2 - margin2
SINF_upper2 = mean2 + margin2

SINF_range2 = SINF_lower2 -SINF_upper2

SINF_lower3 = mean3 - margin3
SINF_upper3 = mean3 + margin3

SINF_range3 = SINF_lower3 -SINF_upper3

SINF_lower4 = mean4 - margin4
SINF_upper4 = mean4 + margin4

SINF_range4 = SINF_lower4 -SINF_upper4

SINF_lower5 = mean5 - margin5
SINF_upper5 = mean5 + margin5

SINF_range5 = SINF_lower5 -SINF_upper5

SINF_lower6 = mean6 - margin6
SINF_upper6 = mean6 + margin6

SINF_range6 = SINF_lower6 -SINF_upper6



print('------------SINF-------------------')


print(np.mean(-SINF_range1),np.max(-SINF_range1))
print(np.mean(-SINF_range2),np.max(-SINF_range2))
print(np.mean(-SINF_range3),np.max(-SINF_range3))
print(np.mean(-SINF_range4),np.max(-SINF_range4))
print(np.mean(-SINF_range5),np.max(-SINF_range5))
print(np.mean(-SINF_range6),np.max(-SINF_range6))


print('-------------------------------------')


# ============================================================
# PLOT
# ============================================================

fig, ax = plt.subplots(figsize=(7, 5))

# Mean
ax.plot(
    x,
    mean1,
    marker='o',
    linewidth=1.8,
    markersize=5,
    label='Uniform'
)

# Confidence interval
ax.fill_between(
    x,
    lower1,
    upper1,
    alpha=0.25,
    # label='95% confidence interval'
)

# Mean
ax.plot(
    x,
    mean2,
    marker='o',
    linewidth=1.8,
    markersize=5,
    label='Beta(6,6)'
)

# Confidence interval
ax.fill_between(
    x,
    lower2,
    upper2,
    alpha=0.25,
    # label='95% confidence interval'
)

# Mean
ax.plot(
    x,
    mean3,
    marker='o',
    linewidth=1.8,
    markersize=5,
    label='Beta(2,6)'
)

# Confidence interval
ax.fill_between(
    x,
    lower3,
    upper3,
    alpha=0.25,
    # label='95% confidence interval'
)

# Mean
ax.plot(
    x,
    mean4,
    marker='o',
    linewidth=1.8,
    markersize=5,
    label='Beta(6,2)'
)

# Confidence interval
ax.fill_between(
    x,
    lower4,
    upper4,
    alpha=0.25,
    # label='95% confidence interval'
)

# Mean
ax.plot(
    x,
    mean5,
    marker='o',
    linewidth=1.8,
    markersize=5,
    label='Quadratic'
)

# Confidence interval
ax.fill_between(
    x,
    lower5,
    upper5,
    alpha=0.25,
    # label='95% confidence interval'
)

# Mean
ax.plot(
    x,
    mean6,
    marker='o',
    linewidth=1.8,
    markersize=5,
    label='Double Peak'
)

# Confidence interval
ax.fill_between(
    x,
    lower6,
    upper6,
    alpha=0.25,
    # label='95% confidence interval'
)


# ============================================================
# AXIS LABELS AND STYLE
# ============================================================

ax.set_xlabel(r'Sample size', fontsize=12)
ax.set_ylabel(r'Sliced Sinf', fontsize=12)

ax.grid(True, alpha=0.3)
ax.legend()

fig.tight_layout()


# ============================================================
# SAVE AS PDF
# ============================================================

fig.savefig(
    'SINF_slsp_log_dip.pdf',
    format='pdf',
    bbox_inches='tight'
)

plt.show()


# ============================================================
# INPUT DATA
# ============================================================





data1 = np.asarray(SUP_dip_gamma)
data2 = np.asarray(SBP_dip_gamma)
data3 = np.asarray(SB26P_dip_gamma)
data4 = np.asarray(SB62P_dip_gamma)
data5 = np.asarray(SQP_dip_gamma)
data6 = np.asarray(SDPP_dip_gamma)


# ============================================================
# X-AXIS VALUES
# ============================================================

# If i simply corresponds to 0, 1, 2, ..., use:
x =[25, 50, 75, 100, 125, 150, 175, 200, 225, 250]




# ============================================================
# COMPUTE MEAN AND 95% CONFIDENCE INTERVAL
# ============================================================

# Mean over the aleatory realizations (axis j)
mean1 = np.mean(data1, axis=1)
mean2 = np.mean(data2, axis=1)
mean3 = np.mean(data3, axis=1)
mean4 = np.mean(data4, axis=1)
mean5 = np.mean(data5, axis=1)
mean6 = np.mean(data6, axis=1)

# Number of realizations for each experiment
n = data1.shape[1]

# Standard deviation over j
std1 = np.std(data1, axis=1, ddof=1)
std2 = np.std(data2, axis=1, ddof=1)
std3 = np.std(data3, axis=1, ddof=1)
std4 = np.std(data4, axis=1, ddof=1)
std5 = np.std(data5, axis=1, ddof=1)
std6 = np.std(data6, axis=1, ddof=1)

# Standard error of the mean
sem1 = std1 / np.sqrt(n)
sem2 = std2 / np.sqrt(n)
sem3 = std3 / np.sqrt(n)
sem4 = std4 / np.sqrt(n)
sem5 = std5 / np.sqrt(n)
sem6 = std6 / np.sqrt(n)

# 95% confidence interval using Student's t distribution
confidence_level = 0.95
alpha = 1.0 - confidence_level

# t_critical = stats.t.ppf(1.0 - alpha / 2.0, df=n - 1)
z_critical = stats.norm.ppf(1.0 - alpha / 2.0)


margin1 = z_critical * sem1
margin2 = z_critical * sem2
margin3 = z_critical * sem3
margin4 = z_critical * sem4
margin5 = z_critical * sem5
margin6 = z_critical * sem6

lower1 = mean1 - margin1
upper1 = mean1 + margin1

lower2 = mean2 - margin2
upper2 = mean2 + margin2

lower3 = mean3 - margin3
upper3 = mean3 + margin3

lower4 = mean4 - margin4
upper4 = mean4 + margin4

lower5 = mean5 - margin5
upper5 = mean5 + margin5

lower6 = mean6 - margin6
upper6 = mean6 + margin6


# ============================================================
# PLOT
# ============================================================

fig, ax = plt.subplots(figsize=(7, 5))

# Mean
ax.plot(
    x,
    mean1,
    marker='o',
    linewidth=1.8,
    markersize=5,
    label='Uniform'
)

# Confidence interval
ax.fill_between(
    x,
    lower1,
    upper1,
    alpha=0.25,
    # label='95% confidence interval'
)

# Mean
ax.plot(
    x,
    mean2,
    marker='o',
    linewidth=1.8,
    markersize=5,
    label='Beta(6,6)'
)

# Confidence interval
ax.fill_between(
    x,
    lower2,
    upper2,
    alpha=0.25,
    # label='95% confidence interval'
)

# Mean
ax.plot(
    x,
    mean3,
    marker='o',
    linewidth=1.8,
    markersize=5,
    label='Beta(2,6)'
)

# Confidence interval
ax.fill_between(
    x,
    lower3,
    upper3,
    alpha=0.25,
    # label='95% confidence interval'
)

# Mean
ax.plot(
    x,
    mean4,
    marker='o',
    linewidth=1.8,
    markersize=5,
    label='Beta(6,2)'
)

# Confidence interval
ax.fill_between(
    x,
    lower4,
    upper4,
    alpha=0.25,
    # label='95% confidence interval'
)

# Mean
ax.plot(
    x,
    mean5,
    marker='o',
    linewidth=1.8,
    markersize=5,
    label='Quadratic'
)

# Confidence interval
ax.fill_between(
    x,
    lower5,
    upper5,
    alpha=0.25,
    # label='95% confidence interval'
)

# Mean
ax.plot(
    x,
    mean6,
    marker='o',
    linewidth=1.8,
    markersize=5,
    label='Double Peak'
)

# Confidence interval
ax.fill_between(
    x,
    lower6,
    upper6,
    alpha=0.25,
    # label='95% confidence interval'
)


# ============================================================
# AXIS LABELS AND STYLE
# ============================================================

ax.set_xlabel(r'Sample size', fontsize=12)
ax.set_ylabel(r'Sliced Sinf', fontsize=12)

ax.grid(True, alpha=0.3)
ax.legend()

fig.tight_layout()


# ============================================================
# SAVE AS PDF
# ============================================================

fig.savefig(
    'SINF_slsp_gamma_dip.pdf',
    format='pdf',
    bbox_inches='tight'
)

plt.show()



# ============================================================
# INPUT DATA
# ============================================================


data1 = np.asarray(SUSinf_indip_gamma)
data2 = np.asarray(SBSinf_indip_gamma)
data3 = np.asarray(SB26Sinf_indip_gamma)
data4 = np.asarray(SB62Sinf_indip_gamma)
data5 = np.asarray(SQSinf_indip_gamma)
data6 = np.asarray(SDPSinf_indip_gamma)


# ============================================================
# X-AXIS VALUES
# ============================================================

# 
x =[25, 50, 75, 100, 125, 150, 175, 200, 225, 250]




# ============================================================
# COMPUTE MEAN AND 95% CONFIDENCE INTERVAL
# ============================================================

# Mean over the aleatory realizations (axis j)
mean1 = np.mean(data1, axis=1)
mean2 = np.mean(data2, axis=1)
mean3 = np.mean(data3, axis=1)
mean4 = np.mean(data4, axis=1)
mean5 = np.mean(data5, axis=1)
mean6 = np.mean(data6, axis=1)

# Number of realizations for each experiment
n = data1.shape[1]

# Standard deviation over j
std1 = np.std(data1, axis=1, ddof=1)
std2 = np.std(data2, axis=1, ddof=1)
std3 = np.std(data3, axis=1, ddof=1)
std4 = np.std(data4, axis=1, ddof=1)
std5 = np.std(data5, axis=1, ddof=1)
std6 = np.std(data6, axis=1, ddof=1)

# Standard error of the mean
sem1 = std1 / np.sqrt(n)
sem2 = std2 / np.sqrt(n)
sem3 = std3 / np.sqrt(n)
sem4 = std4 / np.sqrt(n)
sem5 = std5 / np.sqrt(n)
sem6 = std6 / np.sqrt(n)

# 95% confidence interval using Student's t distribution
confidence_level = 0.95
alpha = 1.0 - confidence_level

# t_critical = stats.t.ppf(1.0 - alpha / 2.0, df=n - 1)

z_critical = stats.norm.ppf(1.0 - alpha / 2.0)


margin1 = z_critical * sem1
margin2 = z_critical * sem2
margin3 = z_critical * sem3
margin4 = z_critical * sem4
margin5 = z_critical * sem5
margin6 = z_critical * sem6

lower1 = mean1 - margin1
upper1 = mean1 + margin1

lower2 = mean2 - margin2
upper2 = mean2 + margin2

lower3 = mean3 - margin3
upper3 = mean3 + margin3

lower4 = mean4 - margin4
upper4 = mean4 + margin4

lower5 = mean5 - margin5
upper5 = mean5 + margin5

lower6 = mean6 - margin6
upper6 = mean6 + margin6


# ============================================================
# PLOT
# ============================================================

fig, ax = plt.subplots(figsize=(7, 5))

# Mean
ax.plot(
    x,
    mean1,
    marker='o',
    linewidth=1.8,
    markersize=5,
    label='Uniform'
)

# Confidence interval
ax.fill_between(
    x,
    lower1,
    upper1,
    alpha=0.25,
    # label='95% confidence interval'
)

# Mean
ax.plot(
    x,
    mean2,
    marker='o',
    linewidth=1.8,
    markersize=5,
    label='Beta(6,6)'
)

# Confidence interval
ax.fill_between(
    x,
    lower2,
    upper2,
    alpha=0.25,
    # label='95% confidence interval'
)

# Mean
ax.plot(
    x,
    mean3,
    marker='o',
    linewidth=1.8,
    markersize=5,
    label='Beta(2,6)'
)

# Confidence interval
ax.fill_between(
    x,
    lower3,
    upper3,
    alpha=0.25,
    # label='95% confidence interval'
)

# Mean
ax.plot(
    x,
    mean4,
    marker='o',
    linewidth=1.8,
    markersize=5,
    label='Beta(6,2)'
)

# Confidence interval
ax.fill_between(
    x,
    lower4,
    upper4,
    alpha=0.25,
    # label='95% confidence interval'
)

# Mean
ax.plot(
    x,
    mean5,
    marker='o',
    linewidth=1.8,
    markersize=5,
    label='Quadratic'
)

# Confidence interval
ax.fill_between(
    x,
    lower5,
    upper5,
    alpha=0.25,
    # label='95% confidence interval'
)

# Mean
ax.plot(
    x,
    mean6,
    marker='o',
    linewidth=1.8,
    markersize=5,
    label='Double Peak'
)

# Confidence interval
ax.fill_between(
    x,
    lower6,
    upper6,
    alpha=0.25,
    # label='95% confidence interval'
)


# ============================================================
# AXIS LABELS AND STYLE
# ============================================================

ax.set_xlabel(r'Sample size', fontsize=12)
ax.set_ylabel(r'Sliced Sinf', fontsize=12)

ax.grid(True, alpha=0.3)
ax.legend()

fig.tight_layout()


# ============================================================
# SAVE AS PDF
# ============================================================

fig.savefig(
    'SINF_slsp_gamma_indip.pdf',
    format='pdf',
    bbox_inches='tight'
)

plt.show()


#%%


# # # -------------------------------------------------------
# # #  BERNOULLI EXPERIMENT FOR DIFFERENT P
# # # -------------------------------------------------------


#
N_sample = 250

AVG_num = 100

P_vals = [0.01,0.13,0.25,0.37,0.49,0.61,0.73,0.85,0.97]

SUG_bern = np.zeros([len(P_vals),AVG_num])
SBG_bern = np.zeros([len(P_vals),AVG_num])
SB62G_bern = np.zeros([len(P_vals),AVG_num])
SB26G_bern = np.zeros([len(P_vals),AVG_num])
SQG_bern = np.zeros([len(P_vals),AVG_num])
SDPG_bern = np.zeros([len(P_vals),AVG_num])



SUP_bern = np.zeros([len(P_vals),AVG_num])
SBP_bern = np.zeros([len(P_vals),AVG_num])
SB62P_bern = np.zeros([len(P_vals),AVG_num])
SB26P_bern = np.zeros([len(P_vals),AVG_num])
SQP_bern = np.zeros([len(P_vals),AVG_num])
SDPP_bern = np.zeros([len(P_vals),AVG_num])

SUS_bern = np.zeros([len(P_vals),AVG_num])
SBS_bern = np.zeros([len(P_vals),AVG_num])
SB62S_bern = np.zeros([len(P_vals),AVG_num])
SB26S_bern = np.zeros([len(P_vals),AVG_num])
SQS_bern = np.zeros([len(P_vals),AVG_num])
SDPS_bern = np.zeros([len(P_vals),AVG_num])

SUL2L1_bern = np.zeros([len(P_vals),AVG_num])
SBL2L1_bern = np.zeros([len(P_vals),AVG_num])
SB62L2L1_bern = np.zeros([len(P_vals),AVG_num])
SB26L2L1_bern = np.zeros([len(P_vals),AVG_num])
SQL2L1_bern = np.zeros([len(P_vals),AVG_num])
SDPL2L1_bern = np.zeros([len(P_vals),AVG_num])

for kk in range(len(P_vals)):
    
    p = P_vals[kk]
    
   
        
    print(p)
    
    

    for jj in range(AVG_num):
        
        X = sample_X(N_sample, 4*(1-p)/5, (1-p)/10, (1-p)/10, seed=None)
    
        SUG_bern[kk,jj] = Sliced_unif_Gini(X) 
        
        
        SUP_bern[kk,jj] = Sliced_unif_Pietra(X) 
       
        
        SUS_bern[kk,jj] = Sliced_unif_Sinf(X) 
        
            
            
#%%
            
x = [0.01,0.13,0.25,0.37,0.49,0.61,0.73,0.85,0.97]



data1 = np.asarray(SUG_bern)
data2 = np.asarray(SUP_bern)
data3 = np.asarray(SUS_bern)

# ============================================================
# COMPUTE MEAN AND 95% CONFIDENCE INTERVAL
# ============================================================

# Mean over the aleatory realizations (axis j)
mean1 = np.mean(data1, axis=1)
mean2 = np.mean(data2, axis=1)
mean3 = np.mean(data3, axis=1)


# Number of realizations for each experiment
n = data1.shape[1]

print(n)

# Standard deviation over j
std1 = np.std(data1, axis=1, ddof=1)
std2 = np.std(data2, axis=1, ddof=1)
std3 = np.std(data3, axis=1, ddof=1)


# Standard error of the mean
sem1 = std1 / np.sqrt(n)
sem2 = std2 / np.sqrt(n)
sem3 = std3 / np.sqrt(n)


# 95% confidence interval using Student's t distribution
confidence_level = 0.95
alpha = 1.0 - confidence_level

# t_critical = stats.t.ppf(1.0 - alpha / 2.0, df=n - 1)

z_critical = stats.norm.ppf(1.0 - alpha / 2.0)

margin1 = z_critical * sem1
margin2 = z_critical * sem2
margin3 = z_critical * sem3


lower1 = mean1 - margin1
upper1 = mean1 + margin1

lower2 = mean2 - margin2
upper2 = mean2 + margin2

lower3 = mean3 - margin3
upper3 = mean3 + margin3

print(lower1,upper1)
print(lower2,upper2)
print(lower3,upper3)


#%%

# ============================================================
# PLOT
# ============================================================

fig, ax = plt.subplots(figsize=(7, 5))

# Mean
ax.plot(
    x,
    mean1,
    marker='o',
    linewidth=1.8,
    markersize=5,
    label='Gini'
)

# Confidence interval
ax.fill_between(
    x,
    lower1,
    upper1,
    alpha=0.25,
    # label='95% confidence interval'
)

# Mean
ax.plot(
    x,
    mean2,
    marker='o',
    linewidth=1.8,
    markersize=5,
    label='Pietra'
)

# Confidence interval
ax.fill_between(
    x,
    lower2,
    upper2,
    alpha=0.25,
    # label='95% confidence interval'
)

# Mean
ax.plot(
    x,
    mean3,
    marker='o',
    linewidth=1.8,
    markersize=5,
    label='Sinf'
)

ax.fill_between(
    x,
    lower3,
    upper3,
    alpha=0.25,
    # label='95% confidence interval'
)


# ============================================================
# AXIS LABELS AND STYLE
# ============================================================

ax.set_xlabel(r'Probability p', fontsize=12)
ax.set_ylabel(r'$Sparisty$', fontsize=12)

ax.grid(True, alpha=0.3)
ax.legend()

fig.tight_layout()


# ============================================================
# SAVE AS PDF
# ============================================================

fig.savefig(
    'my_plot_CI.pdf',
    format='pdf',
    bbox_inches='tight'
)

plt.show()


        

            
            
