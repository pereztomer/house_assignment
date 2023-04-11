from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from glob import glob
from tqdm import tqdm
import json
from hs_utils import extract_points_from_im, avg_noise


def calc_line_features(path):
    x_values, y_values = extract_points_from_im(path)
    # x_values, y_values = avg_noise(x_values, y_values)
    # image = (Image.open("data/line/line_00000.png"))[:,:,0]
    m, b = np.polyfit(x_values, y_values, 1)

    # # Create a plot of the data points and the slope line
    plt.scatter(x_values, y_values)
    plt.plot(x_values, m * x_values + b, color='red')

    # Add labels and title to the plot
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Slope line from a bunch of dots')

    # Show the plot
    plt.show()

    n = len(x_values)
    RSD = np.sqrt(np.sum((y_values - (m * x_values + b)) ** 2) / (n - 2))

    # Calculate the average distance of the points from the regression line
    average_distance = RSD / np.sqrt(n)
    # print("Average distance of the points from the regression line: {:.2f}".format(average_distance))
    return average_distance, m, b


def calc_sin_features(path):
    x, y = extract_points_from_im(path)
    x, y = avg_noise(x, y)
    cords = list(zip(x, y))
    cords.sort(key=lambda a: a[0])
    sorted_x = []
    sorted_y = []

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

    return approx_freq, amplitude
    # # Find the indices of the peaks and troughs
    # peak_indices = np.where(np.diff(np.sign(np.diff(y))) < 0)
    # trough_indices = np.where(np.diff(np.sign(np.diff(y))) > 0)
    #
    # # Calculate the period of the sine wave
    # period = np.mean(np.diff(x[peak_indices]))
    #
    # # Calculate the frequency of the sine wave
    # frequency = 1 / period
    #
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
    # # plt.show()
    #
    # # Print the frequency of the sine wave
    # print("Frequency of the sine wave: {:.2f}".format(frequency))





def calc_curveture(path):
    x, y = extract_points_from_im(path)
    x, y = avg_noise(x, y)
    dydx = np.gradient(y, x)
    # x_t = np.gradient(x)
    # y_t = np.gradient(y)
    # vel = np.array([[x_t[i], y_t[i]] for i in range(x_t.size)])
    # speed = np.sqrt(x_t * x_t + y_t * y_t)
    #
    # tangent = np.array([1 / speed] * 2).transpose() * vel
    # ss_t = np.gradient(speed)
    # xx_t = np.gradient(x_t)
    # yy_t = np.gradient(y_t)
    #
    # curvature_val = np.abs(xx_t * y_t - x_t * yy_t) / (x_t * x_t + y_t * y_t) ** 1.5
    #
    return np.max(dydx), \
        np.min(dydx), \
        np.std(dydx), \
        np.median(dydx), \
        np.percentile(dydx, 90, axis=0), \
        np.percentile(dydx, 10, axis=0)


def calc_parabolic_features(path):
    x, y = extract_points_from_im(path)
    x, y = avg_noise(x, y)
    # Calculate the first derivative
    dydx = np.gradient(y, x)

    # Calculate the second derivative
    d2ydx2 = np.gradient(dydx, x)

    # Calculate the curvature
    curvature = d2ydx2 / (1 + dydx ** 2) ** 1.5

    # #    Create a plot of the parabolic curve and its curvature
    # fig, ax1 = plt.subplots()
    #
    # color = 'tab:red'
    # ax1.set_xlabel('X')
    # ax1.set_ylabel('Y', color=color)
    # ax1.plot(x, y, color=color)
    # ax1.tick_params(axis='y', labelcolor=color)
    #
    # ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
    #
    # color = 'tab:blue'
    # ax2.set_ylabel('Curvature', color=color)  # we already handled the x-label with ax1
    # ax2.plot(x, curvature, color=color)
    # ax2.tick_params(axis='y', labelcolor=color)
    #
    # fig.tight_layout()  # otherwise the right y-label is slightly clipped
    # plt.show()
    return np.max(curvature), \
        np.min(curvature), \
        np.std(curvature), \
        np.median(curvature), \
        np.percentile(curvature, 90, axis=0), \
        np.percentile(curvature, 10, axis=0)


def extract_features():
    files = glob('preprocessed_data/**/*.png', recursive=True)
    ds = []
    labels = {'line': 0, 'parabola': 1, 'sine': 2}
    for f in tqdm(files):
        sample = []
        sample.extend(calc_line_features(f))
        sample.extend(calc_sin_features(f))
        sample.extend(calc_parabolic_features(f))
        sample.extend(calc_curveture(f))
        sample_label = f.split('/')[-2]
        ds.append((sample, labels[sample_label]))

    # Serializing json
    json_object = json.dumps(ds, indent=4)

    # Writing to sample.json
    with open("ds_features.json", "w") as outfile:
        outfile.write(json_object)


def main():
    extract_features()
    # print(calc_curveture(path='preprocessed_data/line/line_00026.png'))
    # print(calc_curveture(path='preprocessed_data/sine/sine_00064.png'))
    # print(calc_parabolic_features(path='preprocessed_data/line/line_00026.png'))
    # exit()
    # calc_line_features(path='preprocessed_data/line/line_00037.png')
    # calc_line_features(path='preprocessed_data/sine/sine_00064.png')
    # print(calc_sin_features(path='data/sine/sine_00064.png'))

    # print(calc_parabolic_features(path='preprocessed_data/line/line_00026.png'))
    # print(calc_parabolic_features(path='preprocessed_data/sine/sine_00064.png'))
    # print(calc_parabolic_features(path='preprocessed_data/parabola/parabola_00101.png'))
    # print(calc_parabolic_features())


if __name__ == '__main__':
    main()
