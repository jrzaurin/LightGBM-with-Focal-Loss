# LightGBM-with-Focal-Loss
An implementation of the focal loss to be used with LightGBM

```python
def focal_loss_lgb(y_pred, dtrain, alpha, gamma):
	"""
	Focal Loss for lightgbm

	Parameters:
	-----------
	y_true: numpy.ndarray
		array with the true labels
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

```