import numpy as np
import lightgbm as lgb
import pickle
import warnings

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from utils.train_hyperopt import LGBOptimizer

warnings.filterwarnings("ignore")


if __name__ == '__main__':

	# ADULT
	databunch = pickle.load(open("data/databunch.p", 'rb'))
	colnames = databunch.colnames
	categorical_columns = databunch.categorical_columns + databunch.crossed_columns
	X = databunch.data
	y = databunch.target
	X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25,
		random_state=1, stratify=y)

	lgbopt = LGBOptimizer(
		[X_tr, y_tr],
		[X_te, y_te],
		colnames,
		categorical_columns,
		with_focal_loss=True)
	lgbopt.optimize(maxevals=50)

	# CREDIT
	databunch = pickle.load(open("data/credit_databunch.p", 'rb'))
	colnames = databunch.colnames
	X = databunch.data
	y = databunch.target
	X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25,
		random_state=1, stratify=y)

	lgbopt = LGBOptimizer(
		[X_tr, y_tr],
		[X_te, y_te],
		colnames,
		with_focal_loss=True)
	lgbopt.optimize(maxevals=50)

