import pandas as pd
from sklearn import linear_model

data = pd.read_csv('BigDataSchool_features.csv', index_col=0, na_values=0)
target = pd.read_csv('BigDataSchool_train_set.csv', index_col=0)
test = pd.read_csv('BigDataSchool_test_set.csv', index_col=0)

X = data
X = X.apply(lambda x: x.fillna(x.mean()), axis=0)
X.dropna(how='all', inplace=True, axis=1)

# X = X.groupby(X.index).apply(lambda x: x.pct_change(-1))
# X.reset_index(level=1, drop=True, inplace=True)
# X.drop(axis=1, labels='MONTH_NUM_FROM_EVENT', inplace=True)
# X = X.iloc[::6, :]

X_tr = X.join(target, how='inner')
X_tst = X.join(test, how='inner')

x_train = X_tr.iloc[:, :-1]
y_train = X_tr.iloc[:, -1:]
x_test = X_tst.iloc[:, :-1]

regr = linear_model.LinearRegression()

# Train the model using the training sets
regr.fit(x_train, y_train)
y_pred = regr.predict(x_test)
q = list(x_test.index)
w = list(map(lambda x: x[0], y_pred.tolist()))

df = pd.DataFrame([*zip(q, w)]).groupby(0).sum()
df.columns = ['TARGET']
df.index.rename('ID', inplace=True)
df.to_csv('kochubeivitalii_test.txt')
