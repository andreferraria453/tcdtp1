import matplotlib.pyplot as plt
def show_elbow_graph(x,y):
    plt.figure(figsize=(8, 5))
    plt.plot(x, y, marker='o')
    plt.title("Elbow Method")
    plt.xlabel("Número de clusters (k)")  # eixo X
    plt.ylabel("Inertia (soma das distâncias)")  # eixo Y
    plt.xticks(x)
    plt.grid(True)
    plt.show()


def show_graph(x,y,title,x_label,y_label):
    plt.figure(figsize=(8, 5))
    plt.plot(x, y, marker='o')
    plt.title(title)
    plt.xlabel(x_label)  # eixo X
    plt.ylabel(y_label)  # eixo Y
    plt.show()

def show_histogram(data_points, title, x_label, y_label):
    plt.figure(figsize=(10, 5))
    plt.hist(data_points, bins=40, color="skyblue", edgecolor="black", alpha=0.7, density=True)
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.grid(True)
    plt.show()