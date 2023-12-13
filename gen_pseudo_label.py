import numpy as np
import statsmodels.api as sm
import math
from scipy.stats import norm


# Return uncertainty threshold, q parameter
def gen_q_func(source_y, max_uncertainty, eta=0.9):
    """
    Generate Q function. This function should be customized, but here is an example of generating a Q function based on
        quantile regression model
    Args:
        source_y: A list of tuple (uncertainty, prediction, ground truth) in source dataset
        max_uncertainty: Any sample with uncertainty larger than it will not be included in the calculation to
            eliminate the influence of outliers
        eta: proportion of the source data used to decide the uncertainty threshold
    Returns:
        Parameters to calculate std, uncertainty threshold
    """
    uncertainty_std_list = []
    for t in source_y:
        if t[0] < max_uncertainty:
            uncertainty_std_list.append([t[0], abs(t[1] - t[2])])
    uncertainty_std_list = np.array(uncertainty_std_list)

    X, Y = uncertainty_std_list[:, 0], uncertainty_std_list[:, 1]
    X = sm.add_constant(X[:, np.newaxis])
    quantreg = sm.QuantReg(Y, X)
    model = quantreg.fit(q=0.6827)
    return model.params, np.quantile(uncertainty_std_list[:, 0], q=eta)


def q_func(uncertainty, q_params):
    """
    Calculate std given an uncertainty. This function should be customized, but here is an example of
        quantile regression model
    Args:
        uncertainty: an uncertainty
        q_params: Parameters of Q function
    Returns:
        STD
    """
    intercept, slope = q_params[0], q_params[1]
    return uncertainty * slope + intercept


def con_classifier(target_y, thresh):
    """
    Split target data into confidence data and uncertain data
    Args:
        target_y: A list of tuple (uncertainty, prediction, sample_id)
        thresh: Uncertainty threshold used to split target data into confidence data and uncertain data
    Returns:
        Set of confidence data, set of uncertain data
    """
    set_c = []
    set_u = []
    for t in target_y:
        if t[0] < thresh:
            set_c.append(t)
        else:
            set_u.append(t)
    return set_c, set_u


def cal_den(et_count, std, minimum, num, size):
    """
    Calculate densities given a point's et_count and std
    Args:
        et_count: estimation
        std: standard deviation
        minimum: minimum value of the density map
        num: number of block in the density map
        size: block size of the density map
    Returns:
        A list of (slot, density)
    """
    def cal_cdf(x, mean, std):
        return norm.cdf((x - mean) / std)
    den_list = []  # to be returned
    sigma_range = [et_count-3*std, et_count+3*std]  # the range [mean-3*sigma, mean+3*sigma] of the point
    partitions = minimum + np.arange(0, num) * size  # left side of the block in the density map
    # partitions in the range
    pos = np.where((partitions >= sigma_range[0]) & (partitions < sigma_range[1]))[0]
    values = partitions[pos]
    if values.shape[0] != 0:
        for i, (p, v) in enumerate(zip(pos, values)):
            if p == 0:
                continue
            elif i == 0:
                den_list.append([p-1, cal_cdf(v, et_count, std)])
            else:
                den_list.append([p-1, cal_cdf(v, et_count, std) - cal_cdf(partitions[p-1], et_count, std)])
        den_list.append([pos[-1], 1-cal_cdf(values[-1], et_count, std)])
    else:
        for i, p in enumerate(partitions):
            if sigma_range[0] >= p:
                den_list.append([i, 1])
                break
    return den_list


def density_map_construct(set_c, q_params, grid_size):
    """
    Generate density map. Here is for the case that prediction is one-dimensional
    Args:
        set_c: set of confidence data
        q_params: parameters for Q function
        grid_size: side length of the grid in the density map
    Returns:
        den_map: density map
        est_map: estimation map, i.e., the corresponding value in each block
        minimum: minimum value of the density map
        num_block: number of block in the density map
    """
    mean_std_list = []  # A list of tuple (mean, std) used to construct density map
    for t in set_c:
        mean_std_list.append((t[1], q_func(t[0], q_params)))

    mean_std_list = np.array(mean_std_list)
    min_data = mean_std_list[:, 0] - 3 * mean_std_list[:, 1]
    max_data = mean_std_list[:, 0] + 3 * mean_std_list[:, 1]
    minimum = np.min(min_data)
    maximum = np.max(max_data)
    num_block = math.ceil((maximum - minimum) / grid_size)

    # Generate density map
    den_map = np.zeros(num_block)
    for data in mean_std_list:
        den_list = cal_den(data[0], data[1], minimum, num_block, grid_size)
        for d in den_list:
            den_map[d[0]] += d[1]
    den_map /= np.sum(den_map)
    return den_map, minimum + grid_size / 2 + np.arange(0, num_block) * grid_size, minimum, num_block


def pseudo_label_gen(den_map, est_map, minimum, num_block, grid_size, set_u):
    """
    Generate pseudo label
    Args:
        den_map: density map generated from confidence data
        est_map: estimation map generated from confidence data
        minimum: minimum value of the side of density map
        num_block: number of grids in the density map
        grid_size: side length of the grid in the density map
        set_u: set of uncertain data
    Returns:
        Pseudo labels: a dictionary of elements {sample_id: (pseudo_label, variance, local mean density)}
    """
    mean_std_list = []  # A list of tuple (mean, std, sample_id, variance) used to calculate pseudo labels
    for t in set_u:
        mean_std_list.append((t[1], q_func(t[0], q_params), t[2], t[0]))

    # Generate pseudo label
    pseudo_label_dict = {}
    for mean_std in mean_std_list:
        den_list = cal_den(mean_std[0], mean_std[1], minimum, num_block, grid_size)
        pseudo_list = []  # To be used for interpolation
        for d in den_list:
            pseudo_list.append((est_map[d[0]], den_map[d[0]] * d[1], den_map[d[0]]))
        pseudo_array = np.array(pseudo_list)
        pseudo_label = np.average(pseudo_array[:, 0], weights=pseudo_array[:, 1]).item()
        pseudo_label_dict[mean_std[2]] = (pseudo_label, mean_std[3], np.mean(pseudo_array[:, 2]).item())
    return pseudo_label_dict


def generator(target_y, q_params, thresh, grid_size):
    """
    Args:
        target_y: A list of tuple (uncertainty, prediction, sample_id)
        q_params: parameters for Q function
        thresh: Uncertainty threshold used to split target data into confidence data and uncertain data
        grid_size: side length of the grid in the density map
    Returns:
        Pseudo labels: a dictionary of elements {sample_id: (pseudo_label, variance, local mean density)}
    """
    set_c, set_u = con_classifier(target_y, thresh)
    den_map, est_map, minimum, num_block = density_map_construct(set_c, q_params)
    pseudo_y = pseudo_label_gen(den_map, est_map, minimum, num_block, grid_size, set_u)
    return pseudo_y

def eval():
    print("Prediction accuracy: ")
    print("Pseudo-label accuracy: ")
    pass

# Pseudo-label generation using housing-price prediction dataset
if __name__ == "__main__":
    # Uncertainty ratio: eta of source uncertainty are less than the uncertainty threshold 
    eta = 0.9
    # Information from source data, which is for building label density map of target scenario 
    # A list of tuple: [(uncertainty, prediction, ground truth)]
    source_y = []
    # Information from target data
    # A list of tuple: [(uncertainty, prediction, sample_id)]
    target_y = []
    # For evaluation
    target_label = []

    max_uncertainty = 0.2  # The largest uncertainty that the task considers
    grid_size = 1.0  # Grid size of label density map, which is a task-dependent parameter
    q_params, u_thresh = gen_q_func(source_y, max_uncertainty, eta=eta)
    # Pseudo label for target data
    pseudo_y = generator(target_y, q_params, u_thresh, grid_size)

    if target_label:
        eval(target_y, pseudo_y, target_label)