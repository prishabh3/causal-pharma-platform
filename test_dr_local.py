import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.model_selection import KFold, cross_val_predict
from xgboost import XGBRegressor

URL = 'http://www.nber.org/~rdehejia/data/nsw_dw.dta'
df = pd.read_stata(URL)
Y = df['re78'].values
T = df['treat'].values.astype(int)
X = df.drop(columns=['data_id', 'treat', 're78']).values
n = len(Y)

prop_model = LogisticRegression(max_iter=5000, solver='lbfgs')
e = cross_val_predict(prop_model, X, T, cv=5, method='predict_proba')[:, 1]
e = np.clip(e, 0.05, 0.95)

for model_name, ModClass in [('Linear', LinearRegression), ('XGB', lambda: XGBRegressor(n_estimators=100, max_depth=3))]:
    mu1 = np.zeros(n)
    mu0 = np.zeros(n)
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    for train_idx, test_idx in kf.split(X):
        X_train, Y_train, T_train = X[train_idx], Y[train_idx], T[train_idx]
        X_test = X[test_idx]
        
        mask1 = T_train == 1
        mod1 = ModClass()
        mod1.fit(X_train[mask1], Y_train[mask1])
        mu1[test_idx] = mod1.predict(X_test)
        
        mask0 = T_train == 0
        mod0 = ModClass()
        mod0.fit(X_train[mask0], Y_train[mask0])
        mu0[test_idx] = mod0.predict(X_test)
        
    dr_i = mu1 - mu0 + T * (Y - mu1) / e - (1 - T) * (Y - mu0) / (1 - e)
    print(f"{model_name} DR ATE:", np.mean(dr_i))

