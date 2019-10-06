import numpy as np
import lightgbm as lgb

from sklearn.metrics import f1_score
from scipy.misc import derivative


def sigmoid(x): return 1./(1. +  np.exp(-x))


def best_threshold(y_true, pred_proba, proba_range, verbose=False):
	"""
	Function to find the probability threshold that optimises the f1_score

	Comment: this function is not used in this repo, but I include it in case the
	it useful

	Parameters:
	-----------
	y_true: numpy.ndarray
		array with the true labels
	pred_proba: numpy.ndarray
		array with the predicted probability
	proba_range: numpy.ndarray
		range of probabilities to explore.
		e.g. np.arange(0.1,0.9,0.01)

	Return:
	-----------
	tuple with the optimal threshold and the corresponding f1_score
	"""
	scores = []
	for prob in proba_range:
		pred = [int(p>prob) for p in pred_proba]
		score = f1_score(y_true,pred)
		scores.append(score)
		if verbose:
			print("INFO: prob threshold: {}.  score :{}".format(round(prob,3), round(score,5)))
	best_score = scores[np.argmax(scores)]
	optimal_threshold = proba_range[np.argmax(scores)]
	return (optimal_threshold, best_score)


def focal_loss_lgb(y_pred, dtrain, alpha, gamma):
	"""
	Focal Loss for lightgbm

	Parameters:
	-----------
	y_pred: numpy.ndarray
		array with the predictions
	dtrain: lightgbm.Dataset
	alpha, gamma: float
		See original paper https://arxiv.org/pdf/1708.02002.pdf
	"""
	a,g = alpha, gamma
	y_true = dtrain.label
	def fl(x,t):
		p = 1/(1+np.exp(-x))
		return -( a*t + (1-a)*(1-t) ) * (( 1 - ( t*p + (1-t)*(1-p)) )**g) * ( t*np.log(p)+(1-t)*np.log(1-p) )
	partial_fl = lambda x: fl(x, y_true)
	grad = derivative(partial_fl, y_pred, n=1, dx=1e-6)
	hess = derivative(partial_fl, y_pred, n=2, dx=1e-6)
	return grad, hess


def focal_loss_lgb_eval_error(y_pred, dtrain, alpha, gamma):
	"""
	Adapation of the Focal Loss for lightgbm to be used as evaluation loss

	Parameters:
	-----------
	y_pred: numpy.ndarray
		array with the predictions
	dtrain: lightgbm.Dataset
	alpha, gamma: float
		See original paper https://arxiv.org/pdf/1708.02002.pdf
	"""
	a,g = alpha, gamma
	y_true = dtrain.label
	p = 1/(1+np.exp(-y_pred))
	loss = -( a*y_true + (1-a)*(1-y_true) ) * (( 1 - ( y_true*p + (1-y_true)*(1-p)) )**g) * ( y_true*np.log(p)+(1-y_true)*np.log(1-p) )
	return 'focal_loss', np.mean(loss), False


def lgb_f1_score(preds, lgbDataset):
	"""
	Implementation of the f1 score to be used as evaluation score for lightgbm

	Parameters:
	-----------
	preds: numpy.ndarray
		array with the predictions
	lgbDataset: lightgbm.Dataset
	"""
	binary_preds = [int(p>0.5) for p in preds]
	y_true = lgbDataset.get_label()
	return 'f1', f1_score(y_true, binary_preds), True


def lgb_focal_f1_score(preds, lgbDataset):
	"""
	Adaptation of the implementation of the f1 score to be used as evaluation
	score for lightgbm. The adaptation is required since when using custom losses
	the row prediction needs to passed through a sigmoid to represent a
	probability

	Parameters:
	-----------
	preds: numpy.ndarray
		array with the predictions
	lgbDataset: lightgbm.Dataset
	"""
	preds = sigmoid(preds)
	binary_preds = [int(p>0.5) for p in preds]
	y_true = lgbDataset.get_label()
	return 'f1', f1_score(y_true, binary_preds), True