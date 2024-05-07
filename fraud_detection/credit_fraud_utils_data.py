import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, PolynomialFeatures
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.impute import SimpleImputer
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.tree._export import plot_tree #for plotting the decision tree

all_reports = {}
def load_data():
    train_df = pd.read_csv('train.csv')
    val_df = pd.read_csv('val.csv')
    y_train_df = train_df['Class']
    y_val_df = val_df['Class']
    train_df.drop('Class', inplace=True, axis=1)
    val_df.drop('Class', inplace=True, axis=1)
    return train_df, val_df, y_train_df, y_val_df

def apply_pipeline(train_df, y_train_df, val_df, y_val_df):
    skewed_columns = ['V1', 'V3', 'V11']

    class CustomSkewnessRegulator(BaseEstimator, TransformerMixin):
        def __init__(self, columns):
            self.columns = columns

        def fit(self):
            return self

        def transform(self, X, y=None):
            x_transformed = X.copy()
            for c in self.columns:
                x_transformed[c] = np.log1p(X[c])
            return x_transformed

        def fit_transform(self, X, y=None):
            self.fit()
            return pd.DataFrame(self.transform(X, y), columns=self.columns)

    transformer = ColumnTransformer([
        ('skewness', CustomSkewnessRegulator(skewed_columns), skewed_columns),
    ],
        remainder='passthrough'
    )
    pipeline = Pipeline(steps=[
        ('trans', transformer),
        ('imp', SimpleImputer(strategy='median')),
        ('poly', PolynomialFeatures(degree=2)),
        ('scale', MinMaxScaler()),
    ])
    model = pipeline.fit(train_df)

    new_train_x = pd.DataFrame(model.transform(train_df))
    new_val_x = pd.DataFrame(model.transform(val_df))
    return new_train_x, y_train_df, new_val_x, y_val_df

def do_preds_eval(model, val_df, y_val_df):
    preds = model.predict(val_df)
    report = classification_report(y_val_df, preds)
    return report

def naive_approach_with_poly(train_df, y_train_df, val_df, y_val_df):
    skewed_columns = ['V1', 'V3', 'V11']
    class CustomSkewnessRegulator(BaseEstimator, TransformerMixin):
        def __init__(self, columns):
            self.columns = columns
        def fit(self):
            return self
        def transform(self, X, y=None):
            x_transformed = X.copy()
            for c in self.columns:
                x_transformed[c] = np.log1p(X[c])
            return x_transformed
        def fit_transform(self, X,y=None):
            self.fit()
            return pd.DataFrame(self.transform(X,y), columns=self.columns)

    transformer = ColumnTransformer([
        ('skewness', CustomSkewnessRegulator(skewed_columns), skewed_columns),
    ],
        remainder='passthrough'
    )
    pipeline = Pipeline(steps=[
        ('trans', transformer),
        ('imp', SimpleImputer(strategy='median')),
        ('poly', PolynomialFeatures(degree=2)),
        ('scale', MinMaxScaler()),
        ('model', LogisticRegression(max_iter=1000))
    ])

    model = pipeline.fit(train_df, y_train_df)
    preds = model.predict(val_df)
    report = classification_report(y_val_df, preds)
    all_reports['naive'] = report
    print('With Poly')
    print(report)
    #the data is very biased towards Zeros (imbalanced dataset)

def naive_approach_without_poly(train_df, y_train_df, val_df, y_val_df):
    skewed_columns = ['V1', 'V3', 'V11']
    class CustomSkewnessRegulator(BaseEstimator, TransformerMixin):
        def __init__(self, columns):
            self.columns = columns
        def fit(self):
            return self
        def transform(self, X, y=None):
            x_transformed = X.copy()
            for c in self.columns:
                x_transformed[c] = np.log1p(X[c])
            return x_transformed
        def fit_transform(self, X,y=None):
            self.fit()
            return pd.DataFrame(self.transform(X,y), columns=self.columns)

    transformer = ColumnTransformer([
        ('skewness', CustomSkewnessRegulator(skewed_columns), skewed_columns),
    ],
        remainder='passthrough'
    )
    pipeline = Pipeline(steps=[
        ('trans', transformer),
        ('imp', SimpleImputer(strategy='median')),
        ('scale', MinMaxScaler()),
        ('model', LogisticRegression(max_iter=1000))
    ])

    model = pipeline.fit(train_df, y_train_df)
    preds = model.predict(val_df)
    report = classification_report(y_val_df, preds)
    all_reports['naive'] = report
    print('Without Poly')
    print(report)

def approach2(train_df, y_train_df, val_df, y_val_df, factor):
    #first apply the base pipelne to preprocess data
    train_df, y_train_df, val_df, y_val_df = apply_pipeline(train_df, y_train_df, val_df, y_val_df)

    #try undersampling the majority class
    from imblearn.under_sampling import RandomUnderSampler

    y_zeros, y_ones = y_train_df.value_counts()[0],y_train_df.value_counts()[1]
    rus = RandomUnderSampler(sampling_strategy={0:factor*y_ones})
    train_us, y_train_us = rus.fit_resample(train_df, y_train_df)

    #apply the model
    model = LogisticRegression(max_iter=1000)
    model.fit(train_us, y_train_us)
    preds = model.predict(val_df)
    report = classification_report(y_val_df, preds)
    print(f'UnderSampling with factor {factor}')
    print(report)
def approach3(train_df, y_train_df, val_df, y_val_df):
    train_df, y_train_df, val_df, y_val_df = apply_pipeline(train_df, y_train_df, val_df, y_val_df)

    from imblearn.over_sampling import SMOTE

    smote = SMOTE(sampling_strategy='auto', random_state=42)
    train_os, y_train_os = smote.fit_resample(train_df, y_train_df)

    model = LogisticRegression(max_iter=100)
    model.fit(train_os, y_train_os)
    report = do_preds_eval(model, val_df, y_val_df)
    print(report)
def approach4(train_df, y_train_df, val_df, y_val_df):
    #do cost sensitive learning
    train_df, y_train_df, val_df, y_val_df = apply_pipeline(train_df, y_train_df, val_df, y_val_df)

    from collections import Counter
    cntr = Counter(y_train_df)
    ratio = cntr[0] / cntr[1]

    model = LogisticRegression(solver='lbfgs', class_weight={0:1, 1:ratio})
    model.fit(train_df, y_train_df)
    report = do_preds_eval(model, val_df, y_val_df)
    print(report)


def randomForestClass(train_df, y_train_df, val_df, y_val_df, depth):
    from sklearn.ensemble import RandomForestClassifier

    train_df, y_train_df, val_df, y_val_df = apply_pipeline(train_df, y_train_df, val_df, y_val_df)

    rfc = RandomForestClassifier(max_depth=depth)
    rfc.fit(train_df, y_train_df)
    preds = rfc.predict(val_df)
    report = classification_report(y_val_df, preds)
    print('Random Forest with depth ', depth)
    print(report)

if __name__ == '__main__':
    train_df, val_df, y_train_df, y_val_df = load_data()
    '''naive_approach_without_poly(train_df, y_train_df, val_df, y_val_df)
    print('===========================================')
    naive_approach_with_poly(train_df, y_train_df, val_df, y_val_df)
    print('===========================================')
    #visualize_without_poly_low_dim(train_df, val_df, y_train_df, y_val_df)
    #visualize_with_poly_low_dim(train_df, val_df, y_train_df, y_val_df)
    approach2(train_df, y_train_df, val_df, y_val_df)
    print('===========================================')
    approach3(train_df, y_train_df, val_df, y_val_df)
    print('===========================================')
    approach4(train_df, y_train_df, val_df, y_val_df)'''
    #knn(train_df, val_df, y_train_df, y_val_df)

    '''for factor in range(10,30,5):
        approach2(train_df,y_train_df ,val_df, y_val_df, factor)'''
    randomForestClass(train_df, y_train_df, val_df, y_val_df, 4)

#try seeing the best threshold
#try k-fold for other hyperparameters