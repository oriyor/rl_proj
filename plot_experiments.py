import pickle
import os
import matplotlib.pyplot as plt


RESULTS_PATH = r"C:\Users\catav\Documents\RL-proj\sc_results"
TARGET_PATH = r"C:\Users\catav\AnacondaProjects\RlProject\experiment_plots"

for results_dir in os.listdir(RESULTS_PATH):
    with open(os.path.join(RESULTS_PATH, results_dir, "statistics.pkl"), "rb") as f:
        results = pickle.load(f)
        for values_title, values in results.items():
            plt.scatter(list(range(len(values))), values, label=values_title)
            plt.xlabel("episode")
            plt.ylabel("reward")
            plt.title(results_dir)
        plt.legend()
        plt.savefig(os.path.join(TARGET_PATH, results_dir + ".png"))
        plt.close()