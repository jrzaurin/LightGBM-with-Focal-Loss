'''
Code here taking from: https://github.com/jrzaurin/ml_pipelines/blob/master/utils/feature_tools.py
'''
import pandas as pd
import copy

from sklearn.utils import Bunch

class FeatureTools(object):
	"""Collection of preprocessing methods"""

	@staticmethod
	def num_scaler(df_inp, cols, sc, trained=False):
		"""
		Method to scale numeric columns in a dataframe
		Parameters:
		-----------
		df_inp: Pandas.DataFrame
		cols: List
			List of numeric columns to be scaled
		sc: Scaler object. From sklearn.preprocessing or similar structure
		trained: Boolean
			If True it will only be used to 'transform'
		Returns:
		--------
		df: Pandas.DataFrame
			transformed/normalised dataframe
		sc: trained scaler
		"""
		df = df_inp.copy()
		if not trained:
			df[cols] = sc.fit_transform(df[cols])
		else:
			df[cols] = sc.transform(df[cols])
		return df, sc

	@staticmethod
	def cross_columns(df_inp, x_cols):
		"""
		Method to build crossed columns. These are new columns that are the
		cartesian product of the parent columns.
		Parameters:
		-----------
		df_inp: Pandas.DataFrame
		x_cols: List.
			List of tuples with the columns to cross
			e.g. [('colA', 'colB'),('colC', 'colD')]
		Returns:
		--------
		df: Pandas.DataFrame
			pandas dataframe with the new crossed columns
		colnames: List
			list the new column names
		"""
		df = df_inp.copy()
		colnames = ['_'.join(x_c) for x_c in x_cols]
		crossed_columns = {k:v for k,v in zip(colnames, x_cols)}

		for k, v in crossed_columns.items():
		    df[k] = df[v].apply(lambda x: '-'.join(x), axis=1)

		return df, colnames

	@staticmethod
	def val2idx(df_inp, cols, val_to_idx=None):
		"""
		This is basically a LabelEncoder that returns a dictionary with the
		mapping of the labels.
		Parameters:
		-----------
		df_inp: Pandas.DataFrame
		cols: List
			List of categorical columns to encode
		val_to_idx: Dict
			LabelEncoding dictionary if already exists
		Returns:
		--------
		df: Pandas.DataFrame
			pandas dataframe with the categorical columns encoded
		val_to_idx: Dict
			dictionary with the encoding mappings
		"""
		df = df_inp.copy()
		if not val_to_idx:

			val_types = dict()
			for c in cols:
			    val_types[c] = df[c].unique()

			val_to_idx = dict()
			for k, v in val_types.items():
			    val_to_idx[k] = {o: i for i, o in enumerate(val_types[k])}

		for k, v in val_to_idx.items():
		    df[k] = df[k].apply(lambda x: v[x])

		return df, val_to_idx

	def __call__(self, df_inp, target_col, scale_cols=None, scaler=None,
		categorical_cols=None, x_cols=None):
		"""
		Parameters:
		-----------
		df_inp: Pandas.DataFrame
		target_col: Str
		scale_cols: List
			List of numerical columns to be scaled
		scaler: Scaler. From sklearn.preprocessing or object with the same
		structure
		categorical_cols: List
			List with the categorical columns
		x_cols: List
			List of tuples with the columns to cross
		"""

		df = df_inp.copy()
		databunch = Bunch()

		if scale_cols:
			assert scaler is not None, 'scaler argument is missing'
			databunch.scale_cols = scale_cols
			df, sc = self.num_scaler(df, scale_cols, scaler)
			databunch.scaler = sc
		else:
			databunch.scale_cols = None

		if categorical_cols:
			databunch.categorical_cols = categorical_cols
			if x_cols:
				df, crossed_cols = self.cross_columns(df, x_cols)
				df, encoding_d = self.val2idx(df, categorical_cols+crossed_cols)
				databunch.crossed_cols = crossed_cols
				databunch.encoding_dict = encoding_d
			else:
				df, encoding_d = self.val2idx(df, categorical_cols)
				databunch.crossed_cols = None
				databunch.encoding_dict = encoding_d
		else:
			databunch.encoding_dict = None
			databunch.categorical_cols = None

		databunch.target = df[target_col]
		df.drop(target_col, axis=1, inplace=True)
		databunch.data = df
		databunch.colnames = df.columns.tolist()

		return databunch

