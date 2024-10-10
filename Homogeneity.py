import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import rankdata
from collections import namedtuple

################################################
############### Sub Functions ##################
################################################
# Supporting Functions (all the functions provided in the previous code)
# Data Preprocessing
def __preprocessing(x):
    try:
        if x.index.dtype != 'int64':
            idx = x.index.date.astype('str')
        else:
            idx = np.asarray(range(1, len(x)+1))
    except:
        idx = np.asarray(range(1, len(x)+1))
        
    x = np.asarray(x)
    dim = x.ndim
    
    if dim == 1:
        c = 1
    elif dim == 2:
        (n, c) = x.shape
        if c == 1:
            dim = 1
            x = x.flatten()
    else:
        print('Please check your dataset.')
    return x, c, idx

# Missing Values Analysis
def __missing_values_analysis(x, idx, method = 'skip'):
    if method.lower() == 'skip':
        if x.ndim == 1:
            idx = idx[~np.isnan(x)]
            x = x[~np.isnan(x)]
        else:
            idx = idx[~np.isnan(x).any(axis=1)]
            x = x[~np.isnan(x).any(axis=1)] 
    n = len(x)
    return x, n, idx

# Pettitt test
def __pettitt(x):
    n = len(x)
    r = rankdata(x)
    k = np.arange(n-1)
    s = r.cumsum()[:-1]   
    U = 2 * s - (k + 1) * (n + 1)
    return abs(U).max(), abs(U).argmax() + 1

# SNHT test
def __snht(x):
    n = len(x)
    k = np.arange(1, n)
    s = x.cumsum()[:-1]
    rs = x[::-1].cumsum()[::-1][1:]
    z1 = ((s - k * x.mean()) / x.std(ddof=1)) / k
    z2 = ((rs - k[::-1] * x.mean()) / x.std(ddof=1)) / (n - k)
    T = (k) * z1 ** 2 + (n - k) * z2 ** 2 
    return T.max() , T.argmax() + 1

# Buishad Q statistics test
def __buishand_q(x, alpha=0.05):
    n = len(x)
    k = np.arange(1, n+1)
    S = x.cumsum() - k * x.mean()  
    S_std = S  / x.std()  # sample std
    Q = abs(S_std).max() / np.sqrt(n)
    return Q, abs(S).argmax() + 1

# Buishad range test
def __buishand_range(x, alpha=0.05):
    n = len(x)
    k = np.arange(1, n+1)
    S = x.cumsum() - k * x.mean() 
    S_std = S  / x.std() # should use sample std -> x.std()
    R = (S_std.max() - S_std.min()) / np.sqrt(n)
    return R, abs(S).argmax() + 1

# Buishad likelihood ratio test
def __buishand_lr(x, alpha=0.05):
    n = len(x)
    k = np.arange(1, n+1)
    S = x.cumsum() - k * x.mean()
    V = S[:-1] / (x.std() * (k[:-1] *(n-k[:-1]))**0.5)
    return abs(V).max(), abs(S).argmax() + 1

# Buishad U statistics test
def __buishand_u(x):
    n = len(x)
    k = np.arange(1, n+1)
    S = x.cumsum() - k * x.mean() 
    S_std = S  / x.std() # should use sample std -> x.std()
    U = (S_std[:n-1]**2).sum() / (n * (n + 1))
    return U, abs(S).argmax() + 1

# Von Neumann test
# def __von_neumann(x):
#     """
#     Von Neumann's ratio test for homogeneity in time series.
#     The ratio is based on the difference between successive elements.
#     Returns:
#         N: Von Neumann ratio
#         cp: Change-point index (Not applicable for this test, return None)
#     """
#     n = len(x)
#     diff = np.diff(x)
#     num = np.sum(diff ** 2)
#     den = np.sum(x ** 2)
#     N = num / den if den != 0 else 0  # Prevent division by zero
#     return N, None

# Monte carlo simulation for p-value calculation
def __mc_p_value(func, stat, n, sim): 
    rand_data = np.random.normal(0, 1, [sim, n])
    res = np.asarray(list(map(func, rand_data)))
    p_val = (res[:,0] > stat).sum() / sim
    return p_val

# Mean calculation
def __mean1(x, loc):
    mu = namedtuple('mean',['mu1', 'mu2'])
    mu1 = x[:loc].mean()
    mu2 = x[loc:].mean()
    return mu(mu1, mu2)

def __mean(x):
    # Calculate the mean of the entire dataset
    mu = x.mean()
    return mu

# Homogeneity test
def __test(func, x, alpha, sim):
    x, c, idx = __preprocessing(x)
    x, n, idx = __missing_values_analysis(x, idx, method = 'skip')
    stat, loc = func(x)
    if sim:
        p = __mc_p_value(func, stat, n, sim)
        h = alpha > p
    else:
        p = None
        h = None
    # Determine `mu` based on homogeneity (`h`)
    if h:
        mu = __mean1(x, loc)  # Homogeneity
    else:
        mu = __mean(x)  # Non-Homogeneity
    return h, idx[loc-1], p, stat, mu

# Pettitt's test and other test functions
 
# Example usage to apply Pettitt's test
def pettitt_test(x, alpha = 0.05, sim = 20000):
    """
    This function checks homogeneity test using A. N. Pettitt's (1979) method.
    Input:
        x: a vector (list, numpy array or pandas series) data
        alpha: significance level (default 0.05)
        sim: No. of monte carlo simulation for p-value calculation (default 20000)
    Output:
        h: True (if data is nonhomogeneous) or False (if data is homogeneous)
        cp: probable change-point location index
        p: p-value of the significance test
        U: Maximum of absolute Pettitt's U Statistics
        avg: mean values at before and after change-point
    Examples
    --------
      >>> x = np.random.rand(1000)
      >>> h, cp, p, U, mu = pettitt_test(x, 0.05)
    """
    res = namedtuple('Pettitt_Test', ['h', 'cp', 'p', 'U', 'avg'])
    h, cp, p, U, mu = __test(__pettitt, x, alpha, sim)
    if not sim:
        x, c, idx = __preprocessing(x)
        x, n, idx = __missing_values_analysis(x, idx, method = 'skip')
        p = 2 * np.exp((- 6 * U**2) / (n**3 + n**2))
        h = alpha > p
    return res(h, cp, p, U, mu)

def snht_test(x, alpha = 0.05, sim = 20000):
    """
    This function checks homogeneity test using H. Alexandersson (1986) method.
    Input:
        x: a vector (list, numpy array or pandas series) data
        alpha: significance level (default 0.05)
        sim: No. of monte carlo simulation for p-value calculation (default 20000)
    Output:
        h: True (if data is nonhomogeneous) or False (if data is homogeneous)
        cp: probable change-point location index
        p: p-value of the significance test
        T: Maximum of SNHT T Statistics
        avg: mean values at before and after change-point
    Examples
    --------
      >>> import pyhomogeneity as hg
      >>> x = np.random.rand(1000)
      >>> h, cp, p, T, mu = hg.snht_test(x, 0.05)
    """
    res = namedtuple('SNHT_Test', ['h', 'cp', 'p', 'T', 'avg'])
    h, cp, p, T, mu = __test(__snht, x, alpha, sim)

    return res(h, cp, p, T, mu)

def buishand_q_test(x, alpha = 0.05, sim = 20000):
    """
    This function checks homogeneity test using Buishand's Q statistics method proposed in T. A. Buishand (1982).
    Input:
        x: a vector (list, numpy array or pandas series) data
        alpha: significance level (default 0.05)
        sim: No. of monte carlo simulation for p-value calculation (default 20000)
    Output:
        h: True (if data is nonhomogeneous) or False (if data is homogeneous)
        cp: probable change-point location index
        p: p-value of the significance test
        Q: Maximum of absolute Buishand's Q Statistics divided by squire root of sample size [Q/sqrt(n)]
        avg: mean values at before and after change-point
    Examples
    --------
      >>> import pyhomogeneity as hg
      >>> x = np.random.rand(1000)
      >>> h, cp, p, Q, mu = hg.buishand_q_test(x, 0.05)
    """
    res = namedtuple('Buishand_Q_Test', ['h', 'cp', 'p', 'Q', 'avg'])
    h, cp, p, Q, mu = __test(__buishand_q, x, alpha, sim)

    return res(h, cp, p, Q, mu)

def buishand_range_test(x, alpha = 0.05, sim = 20000):
    """
    This function checks homogeneity test using Buishand's range method proposed in T. A. Buishand (1982).
    Input:
        x: a vector (list, numpy array or pandas series) data
        alpha: significance level (default 0.05)
        sim: No. of monte carlo simulation for p-value calculation (default 20000)
    Output:
        h: True (if data is nonhomogeneous) or False (if data is homogeneous)
        cp: probable change-point location index
        p: p-value of the significance test
        R: Buishand's Q Statistics range divided by squire root of sample size [R/sqrt(n)]
        avg: mean values at before and after change-point
    Examples
    --------
      >>> import pyhomogeneity as hg
      >>> x = np.random.rand(1000)
      >>> h, cp, p, R, mu = hg.buishand_range_test(x, 0.05)
    """
    res = namedtuple('Buishand_Range_Test', ['h', 'cp', 'p', 'R', 'avg'])
    h, cp, p, R, mu = __test(__buishand_range, x, alpha, sim)

    return res(h, cp, p, R, mu)

def buishand_likelihood_ratio_test(x, alpha = 0.05, sim = 20000):
    """
    This function checks homogeneity test using Buishand's likelihood ration method proposed in T. A. Buishand (1984).
    Input:
        x: a vector (list, numpy array or pandas series) data
        alpha: significance level (default 0.05)
        sim: No. of monte carlo simulation for p-value calculation (default 20000)
    Output:
        h: True (if data is nonhomogeneous) or False (if data is homogeneous)
        cp: probable change-point location index
        p: p-value of the significance test
        V: Maximum of absolute Buishand's weighted adjusted partial sum S
        avg: mean values at before and after change-point
    Examples
    --------
      >>> import pyhomogeneity as hg
      >>> x = np.random.rand(1000)
      >>> h, cp, p, V, mu = hg.buishand_range_test(x, 0.05)
    """
    res = namedtuple('Buishand_Likelihood_Ratio_Test', ['h', 'cp', 'p', 'V', 'avg'])
    h, cp, p, V, mu = __test(__buishand_lr, x, alpha, sim)

    return res(h, cp, p, V, mu)

def buishand_u_test(x, alpha = 0.05, sim = 20000):
    """
    This function checks homogeneity test using Buishand's U statistics method method proposed in T. A. Buishand (1984).
    Input:
        x: a vector (list, numpy array or pandas series) data
        alpha: significance level (default 0.05)
        sim: No. of monte carlo simulation for p-value calculation (default 20000)
    Output:
        h: True (if data is nonhomogeneous) or False (if data is homogeneous)
        cp: probable change-point location index
        p: p-value of the significance test
        U: Buishand's U Statistics
        avg: mean values at before and after change-point
    Examples
    --------
      >>> import pyhomogeneity as hg
      >>> x = np.random.rand(1000)
      >>> h, cp, p, U, mu = hg.buishand_u_test(x, 0.05)
    """
    res = namedtuple('Buishand_U_Test', ['h', 'cp', 'p', 'U', 'avg'])
    h, cp, p, U, mu = __test(__buishand_u, x, alpha, sim)

    return res(h, cp, p, U, mu)

# def von_neumann_test(x, alpha=0.05, sim=None):
#     """
#     This function checks homogeneity test using Von Neumann's ratio method.
#     Input:
#         x: a vector (list, numpy array or pandas series) data
#         alpha: significance level (default 0.05)
#         sim: No. of monte carlo simulation (not applicable here, keep as None)
#     Output:
#         h: Always return False as there's no direct homogeneity status in this test
#         cp: Always return None (no change-point in this test)
#         N: Von Neumann's ratio
#     """
#     res = namedtuple('Von_Neumann_Test', ['h', 'cp', 'p', 'N', 'avg'])
#     h, cp, N, avg = __test(__von_neumann, x, alpha, sim)
#     return res(False, None, None, N, avg)

################################################
################ Main Program ##################
################################################

# Read data from Excel file
input_dir = r'E:\CAO HOC_HAI_DUONG\Thesis-Luanvan\Model\Data\Testing_data\WLraw data\MyThuan.xlsx'   #Change the path and name file
output_dir = r'E:\CAO HOC_HAI_DUONG\Thesis-Luanvan\Model\Data\Testing_data\WLraw data\Results\BQT'
df = pd.read_excel(input_dir, sheet_name='Ave')
name_station = 'My Thuan' #Thay đổi tên biểu đồ cũng như kết quả tệp *.xlsx
# Assuming the data is in the 'data' column
time = df['Time']
data = df['Data']

# Convert the list to a pandas Series
data_series = pd.Series(data)
time_series = pd.Series(time) 

# Define similar functions for other tests: snht_test, buishand_q_test, etc.
# Dictionary to map method names to functions
test_methods = {
    'pettitt_test': pettitt_test,
    'snht_test': snht_test,
    'buishand_q_test': buishand_q_test, # BQT phát hiện sự thay đổi trong trung bình của chuỗi thời gian 
    'buishand_range_test': buishand_range_test, #BRT tập trung vào sự thay đổi trong phạm vi biến đổi của chuỗi thời gian
    'buishand_likelihood_ratio_test': buishand_likelihood_ratio_test,
    'buishand_u_test': buishand_u_test
}
#Choose method test
method_name = 'buishand_q_test'  # Choose from the available test methods

################################################
############### Export Results #################
################################################

def perform_and_export_test(method_name, data_series, output_dir):
    try:
        method_func = test_methods.get(method_name)
        if not method_func:
            raise ValueError(f"Test method '{method_name}' is not defined.")
        # Perform the test
        result = method_func(data_series)
        print(result)

        # Prepare the results for export
        output_data = {
            "Description": [
                "Test Statistic",
                "Location of Change Point",
                "p-value",
                "Homogeneity Status",
                "Mean Before Change Point",
                "Mean After Change Point"
            ],
            "Value": [
                result.U if hasattr(result, 'U') else (
                result.T if hasattr(result, 'T') else (
                result.Q if hasattr(result, 'Q') else (
                result.R if hasattr(result, 'R') else (result.V)))),    # Test Statistic
                result.cp,  # Location of Change Point
                result.p,  # p-value
                "Non-homogeneous" if result.h else "Homogeneous",  # Homogeneity Status
                result.avg.mu1 if isinstance(result.avg, tuple) else result.avg,  # Mean Before Change Point
                result.avg.mu2 if isinstance(result.avg, tuple) else "N/A"  # Mean After Change Point
            ]
        }

        # Convert to DataFrame
        output_df = pd.DataFrame(output_data)

        # Generate file name based on method_name
        output_file_path = f"{output_dir}/{name_station}_{method_name}_result.xlsx"

        # Export to Excel
        output_df.to_excel(output_file_path, index=False)
        print(f"Results have been successfully exported to {output_file_path}")
    except Exception as e:
        print(f"An error occurred during the export: {e}")
# Example usage
perform_and_export_test(method_name, data_series, output_dir)

################################################
################ Plot Results ##################
################################################

def plot_results(result, data_series, time_series):
    plt.figure(figsize=(10, 6))
    
    # Plot the original data
    plt.plot(time_series, data_series, marker="o", label="Data", color="blue")

    # If non-homogeneous, plot two means (before and after change point)
    if result.h:
        plt.axvline(x=time_series[result.cp - 1], color='red', linestyle='--', label=f'Change Point ({time[result.cp-1]})')
        plt.hlines(y=result.avg.mu1, xmin=time_series.min(), xmax=time_series[result.cp - 1], color='green', linestyle='--', label=f'Mean Before Change Point ({round(result.avg.mu1, 2)})')
        plt.hlines(y=result.avg.mu2, xmin=time_series[result.cp], xmax=time_series.max(), color='orange', linestyle='--', label=f'Mean After Change Point ({round(result.avg.mu2, 2)})')
    # If homogeneous, plot the overall mean
    else:
        overall_mean = result.avg
        plt.hlines(y=overall_mean, xmin=time_series.min(), xmax=time_series.max(), color='green', linestyle='--', label=f'Mean ({round(overall_mean, 2)})')
    
    # Set x-ticks to be integers and evenly spaced
    plt.xticks(np.arange(int(time_series.min()), int(time_series.max()) + 1, 6))
    plt.title(f"{name_station} Station")
    plt.xlabel("Time")
    plt.ylabel("Data")
    plt.legend()
    plt.grid(True)
    plt.show()
    
# Assuming `result` is obtained from the test function
result = test_methods[method_name](data_series)

# Plot the results
plot_results(result, data_series, time_series)



