import pandas as pd
import numpy as np
from category_encoders import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_squared_error, classification_report, f1_score, root_mean_squared_error
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, FunctionTransformer, PolynomialFeatures
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor, XGBClassifier

df = pd.read_csv('Placement_Data_Full_Class.csv')

placement_important_columns = ['ssc_p', 'hsc_p', 'degree_p', 'workex']
non_related_features = ['ssc_b', 'hsc_b', 'hsc_s', 'sl_no']
numeric_features = ['ssc_p', 'hsc_p', 'degree_p', 'etest_p', 'mba_p']
categorical_features = ['gender', 'degree_t', 'workex', 'specialisation']

salary = df.salary
status = df.status
#dropping non-relevant features and target columns
df.drop('salary', axis=1, inplace=True)
df.drop('status', axis=1, inplace=True)
df.drop(non_related_features, axis=1, inplace=True)
status = status.map({'Placed':1, 'Not Placed':0})
salary.fillna(0, inplace=True)
y = pd.DataFrame({'status':status.values, 'salary' : salary.values})

#splitting data
x_train, x_test, y_train, y_test = train_test_split(df,y, test_size=0.2, random_state=42)

def get_processing_pipeline(x_train):
    transformer = ColumnTransformer([
        # numeric features
        ('encoder', OneHotEncoder(), categorical_features),
        ('scaler', StandardScaler(), numeric_features),
    ],
        remainder='passthrough'
    )

    pipeline = Pipeline([
        ('trans', transformer),
    ])

    pipeline.fit(x_train)
    return pipeline

def train_log_reg_class(x_train, y_train, x_test, y_test):

    pipeline = get_processing_pipeline(x_train)

    trans_x_train = pipeline.fit_transform(x_train)
    model = LogisticRegression()
    model.fit(trans_x_train, y_train)
    train_preds = model.predict(trans_x_train)
    trans_x_test = pipeline.transform(x_test)
    preds = model.predict(trans_x_test)

    report = classification_report(y_test, preds)
    print(report)
    print('===========================')
    return pd.DataFrame(trans_x_train), pd.DataFrame(trans_x_test), pd.Series(train_preds), pd.Series(preds)

def train_rand_for_class(x_train, y_train, x_test, y_test):
    pipeline = get_processing_pipeline(x_train)

    trans_x_train = pipeline.fit_transform(x_train)
    grid = {'max_depth': [3, 4, 5, 6, 7], 'n_estimators': [50, 80, 100, 120, 150, 200, 250, 300]}

    clf = RandomForestClassifier()
    search = GridSearchCV(clf, param_grid=grid, cv=10)
    search.fit(trans_x_train, y_train)
    print(search.best_params_)

    clf = RandomForestClassifier(**search.best_params_)
    clf.fit(trans_x_train, y_train)
    train_preds = clf.predict(trans_x_train)
    trans_x_test = pipeline.transform(x_test)

    preds = clf.predict(trans_x_test)

    report = classification_report(y_test, preds)
    print(report)
    print('===========================')
    return pd.DataFrame(trans_x_train), pd.DataFrame(trans_x_test), pd.Series(train_preds), pd.Series(preds)

def train_linear_reg_reg(x_train, x_test, y_train, y_test):
    model = LinearRegression()
    model.fit(x_train, y_train)

    train_preds = model.predict(x_train)
    test_preds = model.predict(x_test)

    train_mse = root_mean_squared_error(y_train, train_preds)
    test_mse = root_mean_squared_error(y_test, test_preds)

    print('train_rmse: ', train_mse)
    print('test_rmse: ', test_mse)

def train_lasso_lin_reg_reg(x_train, x_test, y_train, y_test):
    rmses_train = []
    rmses_test = []
    grid = {'alpha':[0.01,0.1,1,2,3,5,10,20,30,50,100,1000]}
    '''search = GridSearchCV(model, grid, cv=10)

    search.fit(x_train, y_train)
    best_alpha = search.best_params_['alpha']
    model = Lasso(best_alpha)
    best_alpha=1000
    '''

    '''for alpha in grid['alpha']:
        model = Lasso(alpha=alpha)
        model.fit(x_train, y_train)
        preds_train = model.predict(x_train)
        preds_test = model.predict(x_test)
        rmse_train = root_mean_squared_error(y_train, preds_train)
        rmse_test = root_mean_squared_error(y_test, preds_test)

        rmses_train.append(rmse_train)
        rmses_test.append(rmse_test)
    plt.plot(grid['alpha'], rmses_train, marker='o' ,color='blue')
    plt.plot(grid['alpha'], rmses_test,marker='o' ,color='red')

    plt.show()'''

    best_alpha = 100 #from the plot
    model = Lasso(alpha=best_alpha)

    model.fit(x_train, y_train)
    #Do Feature Selection
    coefs = np.array(model.coef_[1])
    features_subset = np.array(x_train.columns)[np.abs(coefs) > 0]

    new_x_train = x_train[features_subset]
    new_x_test = x_test[features_subset]

    model.fit(new_x_train, y_train)
    train_prds = model.predict(new_x_train)
    test_prds = model.predict(new_x_test)
    rmse_train = root_mean_squared_error(y_train, train_prds)
    rmse_test = root_mean_squared_error(y_test, test_prds)
    print('train_rmse: ', rmse_train)
    print('test_rmse: ', rmse_test)

def train_dec_tree_class(x_train, x_test,y_train, y_test):
    grid = {'max_depth':[3,4,5,6,7], 'max_features':[2,3,4]}
    clf = DecisionTreeRegressor(max_depth=4, max_features=4)
    '''search = GridSearchCV(clf, param_grid=grid, cv=10)
    search.fit(x_train, y_train)
    print(search.best_params_)'''
    clf.fit(x_train, y_train)

    train_preds = clf.predict(x_train)
    test_preds = clf.predict(x_test)

    rmse_train = root_mean_squared_error(y_train, train_preds)
    rmse_test = root_mean_squared_error(y_test, test_preds)

    print('train_rmse: ', rmse_train)
    print('test_rmse: ', rmse_test)

def train_rand_forest_class(x_train, x_test,y_train, y_test):
    grid = {'max_depth':[3,4,5,6,7], 'n_estimators':[50,80,100,120,150,200,250,300]}
    clf = RandomForestRegressor()
    search = GridSearchCV(clf, param_grid=grid, cv=10)
    search.fit(x_train, y_train)
    print(search.best_params_)
    clf = RandomForestRegressor(**search.best_params_)
    clf.fit(x_train, y_train)

    train_preds = clf.predict(x_train)
    test_preds = clf.predict(x_test)

    rmse_train = root_mean_squared_error(y_train, train_preds)
    rmse_test = root_mean_squared_error(y_test, test_preds)

    print('train_rmse: ', rmse_train)
    print('test_rmse: ', rmse_test)

'''==================XGBOOST===================='''
def train_xgboost_classifier(x_train, x_test, y_train, y_test):
    pipeline = get_processing_pipeline(x_train)

    trans_x_train = pd.DataFrame(pipeline.transform(x_train))
    trans_x_test = pd.DataFrame(pipeline.transform(x_test))

    model = XGBClassifier()
    model.fit(trans_x_train, y_train)

    #plot feature importances
    plt.bar(range(len(model.feature_importances_)), list(model.feature_importances_))
    plt.show()
    feature_importances = list(model.feature_importances_)
    feature_importances.sort(reverse=True)
    top_features_importances = feature_importances[:5]
    top_features_indices =[]
    for f in top_features_importances:
        top_features_indices.append(list(model.feature_importances_).index(f))
    trans_x_train = trans_x_train.iloc[:,top_features_indices]
    trans_x_test = trans_x_test.iloc[:,top_features_indices]

    model = XGBClassifier()
    model.fit(trans_x_train, y_train)
    train_preds = model.predict(trans_x_train)
    test_preds = model.predict(trans_x_test)

    report = classification_report(y_test, test_preds)
    print(report)
    return pd.DataFrame(trans_x_train), pd.DataFrame(trans_x_test), pd.Series(train_preds), pd.Series(test_preds)

def train_xgboost_regressor(x_train, x_test, y_train, y_test):
    model = XGBRegressor()
    model.fit(x_train, y_train)

    #plot feature importances
    plt.bar(range(len(model.feature_importances_)), list(model.feature_importances_))
    plt.show()
    feature_importances = list(model.feature_importances_)
    feature_importances.sort(reverse=True)
    top_features_importances = feature_importances[:5]
    top_features_indices = []
    for f in top_features_importances:
        top_features_indices.append(list(model.feature_importances_).index(f))

    x_train = x_train.iloc[:, top_features_indices]
    x_test = x_test.iloc[:, top_features_indices]

    model = XGBRegressor()
    model.fit(trans_x_train, y_train)
    train_preds = model.predict(trans_x_train)
    test_preds = model.predict(trans_x_test)

    rmse_train = root_mean_squared_error(y_train, train_preds)
    rmse_test = root_mean_squared_error(y_test, test_preds)

    print('train_rmse: ', rmse_train)
    print('test_rmse: ', rmse_test)

trans_x_train, trans_x_test, train_preds, test_preds = train_xgboost_classifier(x_train,x_test, y_train['status'],  y_test['status'])
#train_preds = [True for p in train_preds if p == 1 else False]
train_preds = train_preds.map({0:False, 1: True})
test_preds = test_preds.map({0:False, 1: True})


trans_x_train = trans_x_train[train_preds]
trans_x_test = trans_x_test[test_preds]

salary_train = y_train['salary'].reset_index()
salary_test = y_test['salary'].reset_index()

#get rows in salary_train and salary_test which correspond to 1 in train_preds and test_preds
salary_train_from_preds = salary_train[train_preds]
salary_test_from_preds = salary_test[test_preds]


#train_lasso_lin_reg_reg(trans_x_train, trans_x_test, salary_train_from_preds, salary_test_from_preds)
#train_dec_tree_class(trans_x_train, trans_x_test, salary_train_from_preds, salary_test_from_preds)
#train_rand_forest_class(trans_x_train, trans_x_test, salary_train_from_preds, salary_test_from_preds)

train_xgboost_regressor(trans_x_train, trans_x_test, salary_train_from_preds, salary_test_from_preds)