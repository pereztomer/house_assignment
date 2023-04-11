import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from glob import glob
from tqdm import tqdm
import json
from hs_utils import extract_points_from_im, avg_noise


def calc_line_features(path: str) -> (float, float, float):
    """
    calculating the regression line slop, intercept and average distance from the points in the original image
    :param path:
    :return: 3 different features
    """
    x_values, y_values = extract_points_from_im(path)
    x_values, y_values = avg_noise(x_values, y_values)
    m, b = np.polyfit(x_values, y_values, 1)

    # # Create a plot of the data points and the slope line
    # plt.scatter(x_values, y_values)
    # plt.plot(x_values, m * x_values + b, color='red')
    #
    # # Add labels and title to the plot
    # plt.xlabel('X')
    # plt.ylabel('Y')
    # plt.title('Slope line from a bunch of dots')
    #
    # # Show the plot
    # plt.show()

    n = len(x_values)
    RSD = np.sqrt(np.sum((y_values - (m * x_values + b)) ** 2) / (n - 2))

    # Calculate the average distance of the points from the regression line
    average_distance = RSD / np.sqrt(n)
    return average_distance, m, b


def calc_sin_features(path: str) -> (float, float):
    """
    calculating the amplitude and frequency
    :param path:
    :return: 3 different features
    """
    x, y = extract_points_from_im(path)
    x, y = avg_noise(x, y)
    cords = list(zip(x, y))
    cords.sort(key=lambda a: a[0])

    peaks, _ = find_peaks(y, height=0)
    peak_times = []
    for i in peaks:  # notice correction here! No range or len needed.
        peak_times.append(x[i])

    peak_dists = [next - current for (current, next) in
                  zip(peak_times, peak_times[1:])]  # swapped next with current for positive result
    approx_freq = 1 / (sum(peak_dists) / len(peak_dists))  # take the inverse of what you did before
    # print(approx_freq)
    max_value = np.max(y)
    min_value = np.min(y)

    # Calculate the amplitude of the sine wave
    amplitude = (max_value - min_value) / 2

    # # Create a plot of the data points and the sine wave
    # plt.plot(x, y)
    # plt.scatter(x[peak_indices], y[peak_indices], color='red')
    # plt.scatter(x[trough_indices], y[trough_indices], color='green')
    #
    # # Add labels and title to the plot
    # plt.xlabel('X')
    # plt.ylabel('Y')
    # plt.title('Sine wave')
    #
    # # Show the plot
    # plt.show()

    return approx_freq, amplitude


def calc_derivative_values(path: str):
    """
    calculating derivatives values for different percentiles
    :param path: string
    :return:
    """
    x, y = extract_points_from_im(path)
    x, y = avg_noise(x, y)
    dydx = np.gradient(y, x)
    return np.max(dydx), \
        np.min(dydx), \
        np.std(dydx), \
        np.median(dydx), \
        np.percentile(dydx, 90, axis=0), \
        np.percentile(dydx, 10, axis=0)


def calc_parabolic_features(path: str):
    """
    calculating curvature values for different percentiles
    :param path: string
    :return:
    """
    x, y = extract_points_from_im(path)
    x, y = avg_noise(x, y)
    # Calculate the first derivative
    dydx = np.gradient(y, x)

    # Calculate the second derivative
    d2ydx2 = np.gradient(dydx, x)

    # Calculate the curvature
    curvature = d2ydx2 / (1 + dydx ** 2) ** 1.5
    return np.max(curvature), \
        np.min(curvature), \
        np.std(curvature), \
        np.median(curvature), \
        np.percentile(curvature, 90, axis=0), \
        np.percentile(curvature, 10, axis=0)


def extract_features() -> None:
    """
    Extracting from each image several features, writing the results to a json file.
    :return: None
    """
    files = glob('preprocessed_data/**/*.png', recursive=True)
    ds = []
    labels = {'line': 0, 'parabola': 1, 'sine': 2}
    for f in tqdm(files):
        sample = []
        sample.extend(calc_line_features(f))
        sample.extend(calc_sin_features(f))
        sample.extend(calc_parabolic_features(f))
        sample.extend(calc_derivative_values(f))
        sample_label = f.split('/')[-2]
        ds.append((sample, labels[sample_label]))

    json_object = json.dumps(ds, indent=4)
    with open("ds_features.json", "w") as outfile:
        outfile.write(json_object)


def main():
    extract_features()


if __name__ == '__main__':
    main()
