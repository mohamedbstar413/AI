import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler, StandardScaler, OneHotEncoder, OrdinalEncoder, PolynomialFeatures,FunctionTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt



curYear = 2024
curMonth = 4

def load_data():
    df = pd.read_csv('AmesHousing.csv')
    return df

def split_data(df,y, test_size=0.2):
    train_data,test_data,y_train,y_test = train_test_split(df,y, test_size=test_size, shuffle=True, random_state=42)
    return train_data,y_train, test_data,y_test

def getNumZeros(arr):
    return len([x for x in arr if x == 0])

def change_drop(df):
    df.drop('Order', axis=1, inplace=True)#by inspection: NaN values don't affect
    df.drop('Alley', axis=1, inplace=True)#by inspection: NaN values don't affect
    df.drop('Misc Val', axis=1, inplace=True)#by inspection: NaN values don't affect
    df.drop('Misc Feature', axis=1, inplace=True)#by inspection: NaN values don't affect
    df.drop('Fence', axis=1, inplace=True)#by inspection: NaN values don't affect
    df.drop('PID', axis=1, inplace=True)#by inspection: NaN values don't affect
    df.drop('Pool QC', axis=1, inplace=True)#by inspection: NaN values don't affect
    df.drop('Bsmt Full Bath', axis=1, inplace=True)
    df.drop('Bsmt Half Bath', axis=1, inplace=True)
    df.drop('Garage Yr Blt', axis=1, inplace=True)
    df.drop('Garage Cars', axis=1, inplace=True)

    df['Year Built'] = curYear - df['Year Built']
    df['Year Remod/Add'] = curYear - df['Year Remod/Add']
    df['Yr Sold'] = (curYear - df['Yr Sold']) + ((curMonth - df['Mo Sold']) / 12)

    df.drop('Mo Sold', axis=1, inplace=True)  # we don't need it anymore

    return df

def test_lasso_select_features(df_train, y_train, df_test, y_test):
    df_train = change_drop(df_train)
    df_test = change_drop(df_test)

    class ColumnTransformerDF(ColumnTransformer):
        def fit_transform_df(self, X, y=None):
            transformed_data = super().fit_transform(X, y)
            return pd.DataFrame(transformed_data, columns=self.get_feature_names_out())

        def transform_df(self, X):
            transformed_data = super().transform(X)
            return pd.DataFrame(transformed_data, columns=self.get_feature_names_out())

    cont_cols = [
        'Lot Frontage', 'Lot Area', 'Mas Vnr Area', 'BsmtFin SF 1', 'BsmtFin SF 2', 'Bsmt Unf SF', 'Total Bsmt SF',
        '1st Flr SF', '2nd Flr SF', 'Low Qual Fin SF', 'Gr Liv Area', 'Garage Area',
        'Wood Deck SF', 'Open Porch SF', 'Enclosed Porch', '3Ssn Porch', 'Screen Porch',
        'Pool Area', 'Year Built', 'Year Remod/Add', 'Yr Sold'
    ]
    disc_nom_cols = [
        'MS SubClass', 'MS Zoning', 'Street', 'Land Contour', 'Lot Config', 'Neighborhood', 'Condition 1',
        'Condition 2', 'Bldg Type', 'House Style', 'Roof Style', 'Roof Matl', 'Exterior 1st', 'Exterior 2nd',
        'Mas Vnr Type', 'Foundation', 'Heating', 'Central Air', 'Garage Type', 'Sale Condition', 'Sale Type'
    ]

    ordinal_cols = [
        "Exter Qual", 'Exter Cond', 'Bsmt Qual', 'Bsmt Cond', 'Bsmt Exposure', 'Heating QC', 'Kitchen Qual',
        'Fireplace Qu','Garage Qual', 'Garage Cond','Lot Shape','Utilities','Land Slope','BsmtFin Type 1',
        'BsmtFin Type 2','Electrical','Functional','Garage Finish', 'Paved Drive'
    ]

    cont_transformer = Pipeline([
        ('imputer1', SimpleImputer(strategy='mean')),
        ('scale', StandardScaler()),
        #('poly', PolynomialFeatures())
    ])

    nom_transformer = Pipeline([
        ('imputer2', SimpleImputer(strategy='most_frequent')),
        ('ohe', OneHotEncoder(handle_unknown='ignore'))
    ])

    ord_transformer = Pipeline([
        ('imputer_ord', SimpleImputer(strategy='most_frequent')),
        ('ord_enc', OrdinalEncoder(unknown_value=20,encoded_missing_value=20, handle_unknown='use_encoded_value'))
    ])

    full_trans = ColumnTransformerDF([
        ('cont', cont_transformer, cont_cols),
        # ('skewness', FunctionTransformer(np.log1p, feature_names_out=myFeatureNames), cont_cols),
        ('nom', nom_transformer, disc_nom_cols),
        ('ord', ord_transformer, ordinal_cols),

    ],
        remainder='passthrough'
    )

    df_train_trans = full_trans.fit_transform_df(df_train)
    df_train_trans,df_val_trans,y_train, y_val = train_test_split(df_train_trans,y_train, test_size=0.1, shuffle=False, random_state=42)
    df_test_trans = full_trans.transform(df_test)

    alphas = [0.01,0.1,1,10,100,500,1000,10000]

    mses = []
    numZeroCols = []
    for a in alphas:
        model = Lasso(alpha=a)
        model.fit(df_train_trans, y_train)
        preds = model.predict(df_val_trans)
        rmse = np.sqrt(mean_squared_error(preds, y_val))
        mses.append(rmse)
        numZeroCols.append(getNumZeros(model.coef_))
    # visualize alphas against mses
    fig, axes = plt.subplots(2, 1)
    axes = axes.flatten()
    axes[0].scatter(np.log10(alphas), mses)
    axes[1].scatter(np.log10(alphas), numZeroCols)
    axes[0].set_xlabel('Alpha')
    axes[1].set_xlabel('Alpha')
    axes[0].set_ylabel('Rmse')
    axes[1].set_ylabel('Number of zero coefs_')
    plt.tight_layout()
    plt.show()

def get_full_trans(df):
    df = change_drop(df)

    class ColumnTransformerDF(ColumnTransformer):
        def fit_transform_df(self, X, y=None):
            transformed_data = super().fit_transform(X, y)
            return pd.DataFrame(transformed_data, columns=self.get_feature_names_out())

        def transform_df(self, X):
            transformed_data = super().transform(X)
            return pd.DataFrame(transformed_data, columns=self.get_feature_names_out())

    cont_cols = [
        'Lot Frontage', 'Lot Area', 'Mas Vnr Area', 'BsmtFin SF 1', 'BsmtFin SF 2', 'Bsmt Unf SF', 'Total Bsmt SF',
        '1st Flr SF', '2nd Flr SF', 'Low Qual Fin SF', 'Gr Liv Area', 'Garage Area',
        'Wood Deck SF', 'Open Porch SF', 'Enclosed Porch', '3Ssn Porch', 'Screen Porch',
        'Pool Area', 'Year Built', 'Year Remod/Add', 'Yr Sold'
    ]
    disc_nom_cols = [
        'MS SubClass', 'MS Zoning', 'Street', 'Land Contour', 'Lot Config', 'Neighborhood', 'Condition 1',
        'Condition 2', 'Bldg Type', 'House Style', 'Roof Style', 'Roof Matl', 'Exterior 1st', 'Exterior 2nd',
        'Mas Vnr Type', 'Foundation', 'Heating', 'Central Air', 'Garage Type', 'Sale Condition', 'Sale Type'
    ]

    ordinal_cols = [
        "Exter Qual", 'Exter Cond', 'Bsmt Qual', 'Bsmt Cond', 'Bsmt Exposure', 'Heating QC', 'Kitchen Qual',
        'Fireplace Qu','Garage Qual', 'Garage Cond','Lot Shape','Utilities','Land Slope','BsmtFin Type 1',
        'BsmtFin Type 2','Electrical','Functional','Garage Finish', 'Paved Drive'
    ]

    cont_transformer = Pipeline([
        ('imputer1', SimpleImputer(strategy='mean')),
        ('scale', StandardScaler()),
        #('poly', PolynomialFeatures())
    ])

    nom_transformer = Pipeline([
        ('imputer2', SimpleImputer(strategy='most_frequent')),
        ('ohe', OneHotEncoder(handle_unknown='ignore'))
    ])

    ord_transformer = Pipeline([
        ('imputer_ord', SimpleImputer(strategy='most_frequent')),
        ('ord_enc', OrdinalEncoder(unknown_value=20,encoded_missing_value=20, handle_unknown='use_encoded_value'))
    ])

    full_trans = ColumnTransformerDF([
        ('cont', cont_transformer, cont_cols),
        # ('skewness', FunctionTransformer(np.log1p, feature_names_out=myFeatureNames), cont_cols),
        ('nom', nom_transformer, disc_nom_cols),
        ('ord', ord_transformer, ordinal_cols),

    ],
        remainder='passthrough'
    )


    full_trans.fit(df)
    return full_trans

def apply_trans(df, trans):
    return trans.transform_df(df)
def get_included_features(df,y, trans, alpha):
    model = Lasso(alpha=alpha)
    df = apply_trans(df, trans)
    df_cols = list(df.columns)

    model.fit(df,y)
    coefs =  model.coef_

    assert len(coefs) == len(df_cols)

    included_features = []
    for i in range(len(coefs)):
        if(coefs[i] != 0):
            included_features.append(df_cols[i])
    return included_features

if __name__ == '__main__':
    df = load_data()
    y = df[list(df.columns)[-1]]
    df.drop(list(df.columns)[-1], inplace=True, axis=1)
    train_data, y_train, test_data, y_test = split_data(df, y)
    '''
        use test_lasso_select_features function to do feature selection using lasso
    '''
    # test_lasso_select_features(train_data, y_train, test_data, y_test)
    '''
        After Running test_lasso_select_features ==> We Got best Parameters :
            Alpha = 100 and it removes around 150 Features
    '''
    trans = get_full_trans(train_data)
    included_features = get_included_features(train_data, y_train, trans, 100)
    '''
        Now We Use only These included_features in upcoming models 
    '''
    train_trans, test_trans = apply_trans(train_data, trans), apply_trans(test_data, trans)
    train_included_only, test_included_only = train_trans[included_features], test_trans[included_features]
    '''
        Now Evaluate Whatever model we want on train_included_only and test_included_only
    '''

    import models
    models.testRidgeTransTarget(train_included_only, y_train, test_included_only, y_test)
    '''
        #====testRidgeTransTarget Gave The best RMSE====#
    '''