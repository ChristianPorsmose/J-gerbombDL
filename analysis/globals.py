from pathlib import Path

from matplotlib import pyplot as plt

class Output: 
    path : Path = Path("TEMP_DIR")


class Color:
    colors = plt.cm.tab20.colors

class PhaseName:
    name = ""


class Experiments:
    path : Path = Path("experiments_results_temp")