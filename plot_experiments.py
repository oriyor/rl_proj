import pickle
import os
import matplotlib.pyplot as plt

RESULTS_PATH = r"./runs"
TARGET_PATH = r"./experiment_plots"

for dir_name in os.listdir(RESULTS_PATH):
    with open(os.path.join(RESULTS_PATH, dir_name, "statistics.pkl"), "rb") as f:
        results = pickle.load(f)
        for values_title, values in results.items():
            if values_title == "t":
                continue
            plt.plot(results["t"], values, label=values_title)
            plt.xlabel("steps (t)")
            plt.ylabel("reward")
            plt.title(dir_name)
        plt.legend()
        plt.savefig(os.path.join(TARGET_PATH, dir_name + ".png"))
        plt.close()
