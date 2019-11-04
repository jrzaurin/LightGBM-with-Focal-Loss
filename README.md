# LightGBM with Focal Loss
This is implementation of the [Focal
Loss](https://arxiv.org/pdf/1708.02002.pdf)[1] to be used with LightGBM.

The companion Medium post can be found [here](https://medium.com/@jrzaurin/lightgbm-with-the-focal-loss-for-imbalanced-datasets-9836a9ae00ca).

The Focal Loss for
[LightGBM](https://papers.nips.cc/paper/6907-lightgbm-a-highly-efficient-gradient-boosting-decision-tree.pdf)[2]
can be simply coded as:

```python
def focal_loss_lgb(y_pred, dtrain, alpha, gamma):
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

to use it one would need the corresponding evaluation function:

```python
def focal_loss_lgb_eval_error(y_pred, dtrain, alpha, gamma):
	a,g = alpha, gamma
	y_true = dtrain.label
	p = 1/(1+np.exp(-y_pred))
	loss = -( a*y_true + (1-a)*(1-y_true) ) * (( 1 - ( y_true*p + (1-y_true)*(1-p)) )**g) * ( y_true*np.log(p)+(1-y_true)*np.log(1-p) )
	return 'focal_loss', np.mean(loss), False
```

And to use it, simply:

```python
focal_loss = lambda x,y: focal_loss_lgb(x, y, 0.25, 1.)
eval_error = lambda x,y: focal_loss_lgb_eval_error(x, y, 0.25, 1.)
lgbtrain = lgb.Dataset(X_tr, y_tr, free_raw_data=True)
lgbeval = lgb.Dataset(X_val, y_val)
params  = {'learning_rate':0.1, 'num_boost_round':10}
model = lgb.train(params, lgbtrain, valid_sets=[lgbeval], fobj=focal_loss, feval=eval_error )
```

In the `examples` directory you will find more details, including how to use [Hyperopt](https://github.com/hyperopt/hyperopt) in combination with LightGBM and the Focal Loss, or how to adapt the Focal Loss to a multi-class classification problem.

Any comment: jrzaurin@gmail.com

### References:
[1] Tsung-Yi Lin, Priya Goyal, Ross Girshick, Kaiming He, Piotr Dollár. Focal Loss for Dense Object Detection

[2] Guolin Ke, Qi Meng Thomas Finley, et al., 2017. LightGBM: A Highly Efficient Gradient Boosting
Decision Tree
