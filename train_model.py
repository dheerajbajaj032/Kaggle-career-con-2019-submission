import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold, StratifiedKFold
import time
import matplotlib

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import seaborn as sns

# Building Model
n_fold = 5
fold = StratifiedKFold(n_splits=n_fold, shuffle=True, random_state=11)

