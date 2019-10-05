import pandas as pd
import pickle
import warnings

from pathlib import Path
from utils.feature_tools import FeatureTools
from sklearn.preprocessing import MinMaxScaler, RobustScaler

import pdb

warnings.filterwarnings("ignore")

if __name__ == '__main__':

	PATH = Path('data/')
	train_fname = 'adult.data'
	test_fname  = 'adult.test'

	df_tr = pd.read_csv(PATH/train_fname)
	df_te = pd.read_csv(PATH/test_fname)

	adult_df = pd.concat([df_tr, df_te]).sample(frac=1)
	adult_df.drop('fnlwgt', axis=1, inplace=True)

	adult_df['target'] = (adult_df['income_bracket'].apply(lambda x: '>50K' in x)).astype(int)
	adult_df.drop('income_bracket', axis=1, inplace=True)

	categorical_cols = list(adult_df.select_dtypes(include=['object']).columns)
	scale_cols = [c for c in adult_df.columns if c not in categorical_cols+['target']]
	crossed_cols = (['education', 'occupation'], ['native_country', 'occupation'])

	preprocessor = FeatureTools()
	adult_databunch = preprocessor(adult_df, target_col='target', scale_cols=scale_cols,
		scaler=MinMaxScaler(), categorical_cols=categorical_cols, x_cols=crossed_cols)
	pickle.dump(adult_databunch, open(PATH/'adult_databunch.p', "wb"))

	credit_df = pd.read_csv(PATH/'creditcard.csv.zip')
	scale_cols = ['Time', 'Amount']
	preprocessor = FeatureTools()
	credit_databunch = preprocessor(credit_df, target_col='Class', scale_cols=scale_cols,
		scaler=MinMaxScaler())
	pickle.dump(credit_databunch, open(PATH/'credit_databunch.p', "wb"))

