import pickle
import os
import matplotlib.pyplot as plt

RESULTS_PATH = r"./runs"
TARGET_PATH = r"./experiment_plots"

runs_dict = {"default": "normal_run",
             "middle blacked": "no_middle_side",
             "left side blacked": "no_left_side"}
dir_name = runs_dict["middle blacked"]
with open(os.path.join(RESULTS_PATH, dir_name, "statistics.pkl"), "rb") as f:
    results = pickle.load(f)
    t_lst = results["t"]
for run_name, dir_name in runs_dict.items():
    with open(os.path.join(RESULTS_PATH, dir_name, "statistics.pkl"), "rb") as f:
        results = pickle.load(f)
        for values_title, values in results.items():
            if values_title == "t":
                continue
            plt.plot(t_lst, values[:len(t_lst)], label=(run_name + " " + values_title.replace('episode_', '')).replace('_', ' '))

plt.xlabel("steps (t)")
plt.ylabel("mean reward")
plt.title("Bonus - Using Partial Frames")
plt.ticklabel_format(axis='x', style='sci', scilimits=(0, 0))
plt.legend()
plt.savefig(os.path.join(TARGET_PATH, "plot_bonus.png"))
plt.close()
