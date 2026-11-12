import numpy as np
import random
from scipy.stats import gamma, lognorm, norm
import matplotlib.pyplot as plt
import math
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
    #
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
    

    N = len(values)
    
    threshold = 1.0 / N
    
   
    average = sum(values) / N

    below = [x for x in values if x < average]
    count_below = len(below)
    sum_below = sum(below)

    return count_below/N-sum_below/(N*average)


# -------------------------------
# Sliced Indexes
# -------------------------------

def Sliced_Gini(sample,d):
    M=2000
    
    res=0
    
    for k in range(M):
        v=sample_positive_unit_vector(d, kappa=10.0)
        res=res*(k/(k+1)) + (1/(k+1))*gini(np.dot(sample,v))   
    return res

def Sliced_Pietra(sample,d):
    M=2000
    
    res=0
    
    for k in range(M):
        v=sample_positive_unit_vector(d, kappa=10.0)
        # print(np.dot(sample,v))
        res=res*(k/(k+1)) + (1/(k+1))*pietra(np.dot(sample,v))   
    return res

def Sliced_Sinf(sample,d):
    M=2000
    
    res=0
    
    for k in range(M):
        v=sample_positive_unit_vector(d, kappa=10.0)
        res=res*(k/(k+1)) + (1/(k+1))*sinf(np.dot(sample,v))   
    return res

def Sliced_L2L1(sample,d):
    M=2000
    
    res=0
    
    for k in range(M):
        v=sample_positive_unit_vector(d, kappa=10.0)
        res=res*(k/(k+1)) + (1/(k+1))*l2l1(np.dot(sample,v))   
    return res

# -------------------------------
# Other Functions
# -------------------------------


def sample_dependent_distribution(N, d, rho=0.5, seed=None):
    """
    Samples N values from a d-dimensional dependent distribution
    using a Gaussian copula with lognormal marginals.

    Parameters
    ----------
    N : int
        Number of samples
    d : int
        Dimension
    rho : float in (0,1)
        Correlation strength
    seed : int or None

    Returns
    -------
    samples : ndarray of shape (N, d)
    """
    if seed is not None:
        np.random.seed(seed)

    # Correlation matrix
    Sigma = rho * np.ones((d, d)) + (1 - rho) * np.eye(d)
    L = np.linalg.cholesky(Sigma)

    # Correlated Gaussian samples
    Z = np.random.randn(N, d) @ L.T

    # Gaussian → Uniform
    U = norm.cdf(Z)

    # Uniform → Lognormal
    samples = lognorm.ppf(U, s=1.0)

    return samples

def sample_positive_unit_vector(d, kappa=10.0):
    """
    Samples a unit vector in R^d with all positive entries.

    Parameters
    ----------
    d : int
        Dimension
    kappa : float
        Concentration parameter (higher = more aligned)

    Returns
    -------
    v : ndarray of shape (d,)
        Positive unit vector
    """
    # Mean direction (uniform positive direction)
    mu = np.ones(d)
    mu /= np.linalg.norm(mu)

    # Sample from isotropic normal
    v = np.random.randn(d)

    # Bias toward mu
    v = v + kappa * mu

    # Fold into positive orthant
    v = np.abs(v)

    # Normalize
    v /= np.linalg.norm(v)

    return v


def sample_3d_lognormal(n, mean, sigma, seed=None):
    rng = np.random.default_rng(seed)
    return rng.lognormal(mean=mean, sigma=sigma, size=(n, 3))


def stats_S_infin(values):
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
    if not values:
        raise ValueError("Input vector must not be empty.")

    N = len(values)
    half = N // 2

    sorted_vals = sorted(values)

    average = sum(sorted_vals) / N
    sum_smallest_half = sum(sorted_vals[:half])
    sum_largest_half = sum(sorted_vals[-half:])

    return (sum_largest_half - sum_smallest_half)/average





#%%


N_sample = [25, 50, 75, 100, 125, 150, 175, 200, 225, 250]
x        = [25, 50, 75, 100, 125, 150, 175, 200, 225, 250]

dim = [50,100,150,200]

N_iter = 200

Sl_Gini = np.zeros([len(N_sample),len(dim),N_iter])
Sl_Pietra = np.zeros([len(N_sample),len(dim),N_iter])
Sl_Sinf = np.zeros([len(N_sample),len(dim),N_iter])
Sl_L1L2 = np.zeros([len(N_sample),len(dim),N_iter])




for i_d in range(len(dim)):
    
    d=dim[i_d]

    for i_ns in range(len(N_sample)):
        
        N=N_sample[i_ns]
    
        for k in range(N_iter):
            
            sample = sample_dependent_distribution(N, d, rho=0.5, seed=None)
            
            Sl_Gini[i_ns,i_d,k] = Sliced_Gini(sample,d)
            Sl_Pietra[i_ns,i_d,k] = Sliced_Pietra(sample,d)
            Sl_Sinf[i_ns,i_d,k] = Sliced_Sinf(sample,d)
            # Sl_L1L2[i_ns,i_d,k] = Sliced_L2L1(sample,d)
            
            


#%%


for i_d in range(len(dim)):
    
    

    data1 = np.asarray(Sl_Gini[:,i_d,:])
    data2 = np.asarray(Sl_Pietra[:,i_d,:])
    data3 = np.asarray(Sl_Sinf[:,i_d,:])
    # data4 = np.asarray(Sl_L1L2[:,i_d,:])
    
    mean1 = np.mean(data1, axis=1)
    mean2 = np.mean(data2, axis=1)
    mean3 = np.mean(data3, axis=1)
    # mean4 = np.mean(data4, axis=1)
    
    # Number of realizations for each experiment
    n = data1.shape[1]
    
    # Standard deviation over j
    std1 = np.std(data1, axis=1, ddof=1)
    std2 = np.std(data2, axis=1, ddof=1)
    std3 = np.std(data3, axis=1, ddof=1)
    # std4 = np.std(data4, axis=1, ddof=1)
    
    sem1 = std1 / np.sqrt(n)
    sem2 = std2 / np.sqrt(n)
    sem3 = std3 / np.sqrt(n)
    # sem4 = std4 / np.sqrt(n)
    
    # 95% confidence interval using Student's t distribution
    confidence_level = 0.95
    alpha = 1.0 - confidence_level
    
    # t_critical = stats.t.ppf(1.0 - alpha / 2.0, df=n - 1)
    z_critical = stats.norm.ppf(1.0 - alpha / 2.0)
    margin1 = z_critical * sem1
    margin2 = z_critical * sem2
    margin3 = z_critical * sem3
    # margin4 = z_critical * sem4
    
    
    lower1 = mean1 - margin1
    upper1 = mean1 + margin1
    
    lower2 = mean2 - margin2
    upper2 = mean2 + margin2
    
    lower3 = mean3 - margin3
    upper3 = mean3 + margin3
    
    # lower4 = mean4 - margin4
    # upper4 = mean4 + margin4
    
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
    
    # Confidence interval
    ax.fill_between(
        x,
        lower3,
        upper3,
        alpha=0.25,
        # label='95% confidence interval'
    )
    

    
    
    ax.set_xlabel(r'Sample size', fontsize=12)
    ax.set_ylabel(r'$Sparisty$', fontsize=12)

    ax.grid(True, alpha=0.3)
    ax.legend()

    fig.tight_layout()


    # ============================================================
    # SAVE AS PDF
    # ============================================================

    d = dim[i_d]
    fig.savefig(
        f"d{d}.pdf",
        format='pdf',
        bbox_inches='tight'
    )

    plt.show()












