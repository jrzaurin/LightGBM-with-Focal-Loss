import numpy as np
import pandas as pd
import pickle
import lightgbm as lgb
import warnings

from pathlib import Path
from .metrics import (focal_loss_lgb, focal_loss_lgb_eval_error, lgb_f1_score,
	lgb_focal_f1_score, sigmoid)
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
	recall_score, confusion_matrix)
from hyperopt import hp, tpe, fmin, Trials

import pdb

warnings.filterwarnings("ignore")


class LGBOptimizer(object):
	"""Use Hyperopt to optimize LightGBM

	# Arguments (details only when args are not self-explanatory)
		train_set: List
			List with the training dataset e.g. [X_tr, y_tr]
		eval_set: List
			List with the training dataset
	"""
	def __init__(self, train_set, eval_set, colnames, categorical_columns=None,
		out_dir=Path('data'),  with_focal_loss=False, save=False):

		self.PATH = out_dir
		self.save = save
		self.with_focal_loss = with_focal_loss

		self.early_stop_dict = {}

		self.colnames = colnames
		self.categorical_columns = categorical_columns

		self.X_tr, self.y_tr = train_set[0], train_set[1]
		self.X_val, self.y_val = eval_set[0], eval_set[1]
		self.lgtrain = lgb.Dataset(
			self.X_tr, self.y_tr,
			feature_name=self.colnames,
			categorical_feature = self.categorical_columns,
			free_raw_data=False)
		self.lgvalid = self.lgtrain.create_valid(
			self.X_val, self.y_val)

	def optimize(self, maxevals=200):

		param_space = self.hyperparameter_space()
		objective = self.get_objective(self.lgtrain)
		objective.i=0
		trials = Trials()
		best = fmin(fn=objective,
		            space=param_space,
		            algo=tpe.suggest,
		            max_evals=maxevals,
		            trials=trials)
		best['num_boost_round'] = self.early_stop_dict[trials.best_trial['tid']]
		best['num_leaves'] = int(best['num_leaves'])
		best['verbose'] = -1

		if self.with_focal_loss:
			focal_loss = lambda x,y: focal_loss_lgb(x, y, best['alpha'], best['gamma'])
			model = lgb.train(best, self.lgtrain, fobj=focal_loss)
			preds = model.predict(self.X_val)
			preds = sigmoid(preds)
			preds = (preds > 0.5).astype('int')
		else:
			model = lgb.train(best, self.lgtrain)
			preds = model.predict(self.lgvalid.data)
			preds = (preds > 0.5).astype('int')
		print('acc: {:.4f}, f1 score: {:.4f}, precision: {:.4f}, recall: {:.4f}'.format(
			accuracy_score(self.y_val, preds), f1_score(self.y_val, preds),
			precision_score(self.y_val, preds), recall_score(self.y_val, preds)))
		print('confusion_matrix')
		print(confusion_matrix(self.y_val, preds))

		if self.save:
			if self.with_focal_loss:
				model_fname = 'model_fl.p'
				param_fname = 'best_params_fl.p'
			else:
				model_fname = 'model.p'
				param_fname = 'best_params.p'

			pickle.dump(model, open(self.PATH/model_fname, 'wb'))
			pickle.dump(best, open(self.PATH/param_fname, 'wb'))

		self.best = best
		self.model = model

	def get_objective(self, train):

		def objective(params):
			"""
			objective function for lightgbm.
			"""
			# hyperopt casts as float
			params['num_boost_round'] = int(params['num_boost_round'])
			params['num_leaves'] = int(params['num_leaves'])

			# need to be passed as parameter
			params['is_unbalance'] = True
			params['verbose'] = -1
			params['seed'] = 1

			if self.with_focal_loss:
				focal_loss = lambda x,y: focal_loss_lgb(x, y,
					params['alpha'], params['gamma'])
				cv_result = lgb.cv(
					params,
					train,
					num_boost_round=params['num_boost_round'],
					fobj = focal_loss,
					feval = lgb_focal_f1_score,
					nfold=3,
					stratified=True,
					early_stopping_rounds=20)
			else:
				cv_result = lgb.cv(
					params,
					train,
					num_boost_round=params['num_boost_round'],
					metrics='binary_logloss',
					feval = lgb_f1_score,
					nfold=3,
					stratified=True,
					early_stopping_rounds=20)
			self.early_stop_dict[objective.i] = len(cv_result['f1-mean'])
			score = round(cv_result['f1-mean'][-1], 4)
			objective.i+=1
			return -score

		return objective

	def hyperparameter_space(self, param_space=None):

		space = {
			'learning_rate': hp.uniform('learning_rate', 0.01, 0.2),
			'num_boost_round': hp.quniform('num_boost_round', 50, 500, 20),
			'num_leaves': hp.quniform('num_leaves', 31, 255, 4),
		    'min_child_weight': hp.uniform('min_child_weight', 0.1, 10),
		    'colsample_bytree': hp.uniform('colsample_bytree', 0.5, 1.),
		    'subsample': hp.uniform('subsample', 0.5, 1.),
		    'reg_alpha': hp.uniform('reg_alpha', 0.01, 0.1),
		    'reg_lambda': hp.uniform('reg_lambda', 0.01, 0.1),
		}
		if self.with_focal_loss:
			space['alpha'] = hp.uniform('alpha', 0.1, 0.75)
			space['gamma'] = hp.uniform('gamma', 0.5, 5)
		if param_space:
			return param_space
		else:
			return space