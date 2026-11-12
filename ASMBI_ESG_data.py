import numpy as np
import random
from scipy.stats import gamma, lognorm, norm
import matplotlib.pyplot as plt
import math
from pathlib import Path
import pandas as pd

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
# Sliced Indexes
# -------------------------------

def Sliced_Gini(sample,d):
    M=10000
    
    res=0
    
    for k in range(M):
        v=sample_positive_unit_vector(d, kappa=10.0)
        res=res*(k/(k+1)) + (1/(k+1))*gini(np.dot(sample,v))   
    return res

def Sliced_Pietra(sample,d):
    M=10000
    
    res=0
    
    for k in range(M):
        v=sample_positive_unit_vector(d, kappa=10.0)
        # print(np.dot(sample,v))
        res=res*(k/(k+1)) + (1/(k+1))*pietra(np.dot(sample,v))   
    return res

def Sliced_Sinf(sample,d):
    M=10000
    
    res=0
    
    for k in range(M):
        v=sample_positive_unit_vector(d, kappa=10.0)
        res=res*(k/(k+1)) + (1/(k+1))*sinf(np.dot(sample,v))   
    return res

def Sliced_L2L1(sample,d):
    M=10000
    
    res=0
    
    for k in range(M):
        v=sample_positive_unit_vector(d, kappa=10.0)
        res=res*(k/(k+1)) + (1/(k+1))*l2l1(np.dot(sample,v))   
    return res


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

def drop_rows_with_negatives(df):
    """
    Removes rows that contain at least one negative value
    in any numeric column.
    """
    numeric_df = df.select_dtypes(include="number")
    mask = (numeric_df >= 0).all(axis=1)
    return df.loc[mask]




def bootstrap_Sliced_Gini(x, d, n_bootstrap=100, confidence=0.95, random_state=None):
    """
    Bootstrap the value of F on an input array.

    Parameters
    ----------
    x : array-like
        Original data array.
    n_bootstrap : int
        Number of bootstrap samples.
    confidence : float
        Confidence level (e.g. 0.95 for a 95% CI).
    random_state : int or None
        Seed for reproducibility.

    Returns
    -------
    mean_F : float
        Bootstrap average of F.
    ci : tuple
        Lower and upper bounds of the confidence interval.
    """

    perc=0.3
    x = np.asarray(x)
    rng = np.random.default_rng(random_state)

    bootstrap_values = np.empty(n_bootstrap)

    for i in range(n_bootstrap):
        # Sample with replacement
        x_boot = rng.choice(x, size=int(perc*len(x)), replace=False)

        # Evaluate the already-defined function F
        bootstrap_values[i] = Sliced_Gini(x_boot,d)

    # Bootstrap estimate
    mean_F = np.mean(bootstrap_values)
    
    
    se_F = np.std(bootstrap_values, ddof=1)

    # Percentile confidence interval
    # alpha = 1 - confidence
    # lower = np.percentile(bootstrap_values, 100 * alpha / 2)
    # upper = np.percentile(bootstrap_values, 100 * (1 - alpha / 2))
    
    alpha = 1 - confidence
    z = norm.ppf(1 - alpha / 2)

    lower = mean_F - z * se_F
    upper = mean_F + z * se_F

    return mean_F, (lower, upper)


def bootstrap_Sliced_Pietra(x, d, n_bootstrap=100, confidence=0.95, random_state=None):
    """
    Bootstrap the value of F on an input array.

    Parameters
    ----------
    x : array-like
        Original data array.
    n_bootstrap : int
        Number of bootstrap samples.
    confidence : float
        Confidence level (e.g. 0.95 for a 95% CI).
    random_state : int or None
        Seed for reproducibility.

    Returns
    -------
    mean_F : float
        Bootstrap average of F.
    ci : tuple
        Lower and upper bounds of the confidence interval.
    """
    
    
    perc=0.6

    x = np.asarray(x)
    rng = np.random.default_rng(random_state)

    bootstrap_values = np.empty(n_bootstrap)

    for i in range(n_bootstrap):
        # Sample with replacement
        x_boot = rng.choice(x, size=int(perc*len(x)), replace=False)

        # Evaluate the already-defined function F
        bootstrap_values[i] = Sliced_Pietra(x_boot,d)

    # Bootstrap estimate
    mean_F = np.mean(bootstrap_values)
    
    se_F = np.std(bootstrap_values, ddof=1)

    #
    
    alpha = 1 - confidence
    z = norm.ppf(1 - alpha / 2)

    lower = mean_F - z * se_F
    upper = mean_F + z * se_F

    return mean_F, (lower, upper)


def bootstrap_Sliced_Sinf(x, d, n_bootstrap=100, confidence=0.95, random_state=None):
    """
    Bootstrap the value of F on an input array.

    Parameters
    ----------
    x : array-like
        Original data array.
    n_bootstrap : int
        Number of bootstrap samples.
    confidence : float
        Confidence level (e.g. 0.95 for a 95% CI).
    random_state : int or None
        Seed for reproducibility.

    Returns
    -------
    mean_F : float
        Bootstrap average of F.
    ci : tuple
        Lower and upper bounds of the confidence interval.
    """

    perc=0.3
    x = np.asarray(x)
    rng = np.random.default_rng(random_state)

    bootstrap_values = np.empty(n_bootstrap)

    for i in range(n_bootstrap):
        # Sample with replacement
        x_boot = rng.choice(x, size=int(perc*len(x)), replace=False)

        # Evaluate the already-defined function F
        bootstrap_values[i] = Sliced_Sinf(x_boot,d)

    # Bootstrap estimate
    mean_F = np.mean(bootstrap_values)
    
    se_F = np.std(bootstrap_values, ddof=1)

  
    
    alpha = 1 - confidence
    z = norm.ppf(1 - alpha / 2)

    lower = mean_F - z * se_F
    upper = mean_F + z * se_F

    return mean_F, (lower, upper)



def bootstrap_Sliced_L2L1(x, d, n_bootstrap=1000, confidence=0.95, random_state=None):
    """
    Bootstrap the value of F on an input array.

    Parameters
    ----------
    x : array-like
        Original data array.
    n_bootstrap : int
        Number of bootstrap samples.
    confidence : float
        Confidence level (e.g. 0.95 for a 95% CI).
    random_state : int or None
        Seed for reproducibility.

    Returns
    -------
    mean_F : float
        Bootstrap average of F.
    ci : tuple
        Lower and upper bounds of the confidence interval.
    """

    perc=0.3
    x = np.asarray(x)
    rng = np.random.default_rng(random_state)

    bootstrap_values = np.empty(n_bootstrap)

    for i in range(n_bootstrap):
        # Sample with replacement
        x_boot = rng.choice(x, size=int(perc*len(x)), replace=False)

        # Evaluate the already-defined function F
        bootstrap_values[i] = Sliced_L2L1(x_boot,d)

 
    mean_F = np.mean(bootstrap_values)
    
    se_F = np.std(bootstrap_values, ddof=1)

    
    
    alpha = 1 - confidence
    z = norm.ppf(1 - alpha / 2)

    lower = mean_F - z * se_F
    upper = mean_F + z * se_F

    return mean_F, (lower, upper)


#%%


DATA_DIR = Path("./")
path = DATA_DIR / "esg_smes.csv"

dfs_ESG = pd.read_csv(
    path,
    # usecols=[0, column_name],   # company + desired column
    index_col=0
)


dfs_ESG = dfs_ESG.drop('NINC', axis=1)
dfs_ESG = dfs_ESG.drop('EBTD', axis=1)

# print(dfs_ESG)

raw_groups = dict(tuple(dfs_ESG.groupby(dfs_ESG.columns[0])))



arrays = {
    country: (
        df.iloc[:, 1:]
          .astype(str)
          .apply(lambda col: col.str.replace(",", ".", regex=False))
          .astype(float)
          .to_numpy()
    )
    for country, df in raw_groups.items()
}


sparsity_ESG = dict.fromkeys(raw_groups.keys())
sparsity_TST = dict.fromkeys(raw_groups.keys())
sparsity_TOT = dict.fromkeys(raw_groups.keys())

for key in arrays.keys():
    
    
    for k in range(len(arrays[key])):
        
        arrays[key]=arrays[key][(arrays[key] >= 0).all(axis=1)]
    print(len(arrays[key]))
    
for key, mat in arrays.items():
    mat[:, -3:] *= 0.0001
    
for key, mat in arrays.items():
    print(key, mat.max())

ESG_dict = {key: mat[:, 1:4] for key, mat in arrays.items()}
TST_dict = {key:  mat[:, -3:] for key, mat in arrays.items()}

TOT = np.vstack([arrays[k] for k in sorted(arrays)])


#%%

for key in arrays.keys():
      
          
        print('key:', key)
        
        values_ESG = ESG_dict[key]
        values_TST = TST_dict[key]
        values_TOT = arrays[key]
        
        sparsity_TOT[key] = [bootstrap_Sliced_Gini(values_TOT[:, 1:],6), bootstrap_Sliced_Pietra(values_TOT[:, 1:],6), bootstrap_Sliced_Sinf(values_TOT[:, 1:],6), bootstrap_Sliced_L2L1(values_TOT[:, 1:],6)]
        sparsity_TST[key] = [bootstrap_Sliced_Gini(values_TST,3), bootstrap_Sliced_Pietra(values_TST,3), bootstrap_Sliced_Sinf(values_TST,3), bootstrap_Sliced_L2L1(values_TST,3)]
        sparsity_ESG[key] = [bootstrap_Sliced_Gini(values_ESG,3), bootstrap_Sliced_Pietra(values_ESG,3), bootstrap_Sliced_Sinf(values_ESG,3), bootstrap_Sliced_L2L1(values_ESG,3)]
        
        
#%%


print('-----')
print('Total')        
print(bootstrap_Sliced_Gini(TOT[:, 1:],6), bootstrap_Sliced_Pietra(TOT[:, 1:],6), bootstrap_Sliced_Sinf(TOT[:, 1:],6),Sliced_L2L1(TOT[:, 1:],6))
        
print('-----')
print('ESG') 
print(bootstrap_Sliced_Gini(TOT[:, 4:],3), bootstrap_Sliced_Pietra(TOT[:, 4:],3), bootstrap_Sliced_Sinf(TOT[:, 4:],3),Sliced_L2L1(TOT[:, 4:],3))

print('-----')
print('TST')          
print(bootstrap_Sliced_Gini(TOT[:, 1:4],3), bootstrap_Sliced_Pietra(TOT[:, 1:4],3), bootstrap_Sliced_Sinf(TOT[:, 1:4],3),Sliced_L2L1(TOT[:, 1:4],3))   
        
        
        
        
        
        
        