from sklearn.linear_model import Ridge, LinearRegression
from sklearn.model_selection import cross_val_score
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.compose import TransformedTargetRegressor

def trainRidge(x,y):
    alphas = [0.01,0.1,1,10,100,1000,10000]
    scores = []
    for a in alphas:
        model = Ridge(alpha=a)
        score = cross_val_score(model, x,y)
        scores.append(np.mean(score))

    return alphas[np.argmax(scores)], np.max(scores) #returns alpha that get the maximum score
def trainRidgeTargetTrans(x,y):
    alphas = [0.01, 0.1, 1, 10, 100, 1000, 10000]
    scores = []
    for a in alphas:
        tt = TransformedTargetRegressor(
            regressor=Ridge(alpha=a), func=np.log, inverse_func=np.exp
        )
        score = cross_val_score(tt, x, y)
        scores.append(np.mean(score))

    return alphas[np.argmax(scores)], np.max(scores)  # returns alpha that get the maximum score
def testRidge(x_train, y_train, x_test,y_test):
    Alpha, score = trainRidge(x_train,y_train)
    print(f'train alpha={Alpha},train score={score}')
    model = Ridge(alpha=Alpha)
    model.fit(x_train, y_train)
    train_preds = model.predict(x_train)
    rmse_train = np.sqrt(mean_squared_error(train_preds, y_train))
    print(f'train rmse={rmse_train}')
    preds = model.predict(x_test)
    rmse = np.sqrt(mean_squared_error(preds, y_test))
    print(f'test rmse={rmse}')

def testRidgeTransTarget(x_train, y_train, x_test,y_test):
    Alpha, score = trainRidgeTargetTrans(x_train, y_train)
    print(f'train alpha={Alpha},train score={score}')
    targetTransformer = TransformedTargetRegressor(
        regressor=Ridge(alpha=Alpha), func=np.log, inverse_func=np.exp
    )
    targetTransformer.fit(x_train, y_train)
    preds = targetTransformer.predict(x_test)
    rmse = np.sqrt(mean_squared_error(preds, y_test))
    print(preds)
    print(f'test rmse={rmse}')


def testLinearReg(x_train, y_train, x_test,y_test):
    model = LinearRegression()
    model.fit(x_train, y_train)

    preds = model.predict(x_test)
    rmse = np.sqrt(mean_squared_error(preds, y_test))
    print(f'test rmse={rmse}')

def testNN():
    pass