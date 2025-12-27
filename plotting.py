import matplotlib.pyplot as plt



def plot_data(x, y, z, title="", xlabel="", ylabel=""):
    plt.figure(figsize=(10, 6))
    plt.plot(x, y, marker='o')
    plt.plot(x, z, marker='x')
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True)
    plt.tight_layout()
    plt.show()