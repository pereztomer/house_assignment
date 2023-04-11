import numpy as np
import matplotlib.pyplot as plt


def plot_line_with_noise(plot_name, line_slope, noise_std):
    # define the x values
    x = np.linspace(0, 1, 100)

    # define the y values for the line with noise
    y = line_slope * x + np.random.normal(0, noise_std, len(x))

    # create a figure and axis object
    fig, ax = plt.subplots()

    # plot the line with noise as a scatter plot with grayscale color and smaller markers
    ax.scatter(x, y, color='black', s=5, marker='o')

    # remove plot frame
    ax.axis('off')

    # save the plot to a file without any whitespace
    plt.savefig(plot_name, bbox_inches='tight', pad_inches=0)


def generate_lines():
    counter = 0
    for line_slop in np.linspace(-3, 3.0, num=53):
        for std in np.linspace(0.1, 0.4, num=7):
            plot_line_with_noise(f'extended_ds/line/{counter}.png', line_slop, std)
            counter += 1


def plot_parabola_with_noise(plot_name, a, b, c, noise_std):
    # define the x values
    x = np.linspace(-1, 1, 100)

    # define the y values for the parabola with noise
    y = a * x ** 2 + b * x + c + np.random.normal(0, noise_std, len(x))

    # create a figure and axis object
    fig, ax = plt.subplots()

    # plot the parabola with noise as a scatter plot with grayscale color and smaller markers
    ax.scatter(x, y, color='gray', s=5, marker='o')

    # remove plot frame
    ax.axis('off')

    # save the plot to a file without any whitespace
    plt.savefig(plot_name, bbox_inches='tight', pad_inches=0)


def generate_parabole():
    counter = 0
    for a in np.linspace(-3, 3.0, num=5):
        for b in np.linspace(0.01, 0.3, num=5):
            for c in np.linspace(0.01, 0.3, num=5):
                for noise_std in np.linspace(0.1, 0.3, num=3):
                    plot_parabola_with_noise(f'extended_ds/parabola/{counter}.png', a, b, c, noise_std)
                    counter += 1


def plot_sine_with_noise(plot_name, amplitude, frequency, phase, noise_std):
    # define the x values
    x = np.linspace(-np.pi, np.pi, 100)

    # define the y values for the sine function with noise
    y = amplitude * np.sin(frequency * x + phase) + np.random.normal(0, noise_std, len(x))

    # create a figure and axis object
    fig, ax = plt.subplots()

    # plot the sine function with noise as a scatter plot with grayscale color and smaller markers
    ax.scatter(x, y, color='gray', s=5, marker='o')

    # remove plot frame
    ax.axis('off')

    # save the plot to a file without any whitespace
    plt.savefig(plot_name, bbox_inches='tight', pad_inches=0)


def generate_sine():
    counter = 0
    for amplitude in np.linspace(-3, 3.0, num=5):
        for frequency in np.linspace(0.01, 0.3, num=5):
            for phase in np.linspace(0.01, 0.3, num=5):
                for noise_std in np.linspace(0.1, 0.3, num=3):
                    plot_parabola_with_noise(f'extended_ds/sine/{counter}.png', amplitude, frequency, phase, noise_std)
                    counter += 1


def main():
    generate_lines()

if __name__ == '__main__':
    main()
