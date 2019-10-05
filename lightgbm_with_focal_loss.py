import numpy as np
import lightgbm as lgb
import argparse
import pickle
import warnings

from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from utils.train_hyperopt import LGBOptimizer


warnings.filterwarnings("ignore")

if __name__ == '__main__':

	ap = argparse.ArgumentParser()
	ap.add_argument("--dataset", required=True)
	ap.add_argument("--save_experiment", action="store_true")
	ap.add_argument("--with_focal_loss", action="store_true")
	ap.add_argument("--is_unbalance", action="store_true")
	args = vars(ap.parse_args())

	PATH = Path("data/")
	is_unbalance = args['is_unbalance']
	with_focal_loss = args['with_focal_loss']
	save_experiment = args['save_experiment']

	if args['dataset'] == 'adult':

		databunch = pickle.load(open(PATH/'adult_databunch.p', 'rb'))
		colnames = databunch.colnames
		categorical_columns = databunch.categorical_columns + databunch.crossed_columns
		X = databunch.data
		y = databunch.target
		X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25,
			random_state=1, stratify=y)

		lgbopt = LGBOptimizer(
			args['dataset'],
			train_set=[X_tr, y_tr],
			eval_set=[X_te, y_te],
			colnames=colnames,
			categorical_columns=categorical_columns,
			is_unbalance=is_unbalance,
			with_focal_loss=with_focal_loss,
			save=save_experiment)
		lgbopt.optimize(maxevals=100)

	if args['dataset'] == 'credit':

		databunch = pickle.load(open("data/credit_databunch.p", 'rb'))
		colnames = databunch.colnames
		X = databunch.data
		y = databunch.target
		X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25,
			random_state=1, stratify=y)

		lgbopt = LGBOptimizer(
			args['dataset'],
			train_set=[X_tr, y_tr],
			eval_set=[X_te, y_te],
			colnames=colnames,
			is_unbalance=is_unbalance,
			with_focal_loss=with_focal_loss,
			save=save_experiment)
		lgbopt.optimize(maxevals=100)

