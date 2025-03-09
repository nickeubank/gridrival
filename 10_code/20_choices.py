import numpy as np
import pandas as pd

pd.set_option("mode.copy_on_write", True)

picks = pd.read_csv("../40_results/results_2025_03_06_17_41.csv")

picks.head(30)
