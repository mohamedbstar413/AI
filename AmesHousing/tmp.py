def prrprocess2(df):
    class ColumnTransformerDF(ColumnTransformer):
        def fit_transform_df(self, X, y=None):
            transformed_data = super().fit_transform(X, y)
            return pd.DataFrame(transformed_data, columns=self.get_feature_names_out())

        def transform_df(self, X):
            transformed_data = super().transform(X)
            return pd.DataFrame(transformed_data, columns=self.get_feature_names_out())

    df.drop('Order', axis=1, inplace=True)
    df.drop('Alley', axis=1, inplace=True)
    df.drop('Misc Val', axis=1, inplace=True)
    df.drop('Misc Feature', axis=1, inplace=True)
    df.drop('Fence', axis=1, inplace=True)
    df.drop('PID', axis=1, inplace=True)
    df.drop('Pool QC', axis=1, inplace=True)
    df.drop('Bsmt Full Bath', axis=1, inplace=True)
    df.drop('Bsmt Half Bath', axis=1, inplace=True)
    df.drop('Garage Yr Blt', axis=1, inplace=True)
    df.drop('Garage Cars', axis=1, inplace=True)


    df['Year Built'] = curYear - df['Year Built']
    df['Year Remod/Add'] = curYear - df['Year Remod/Add']
    df['Yr Sold'] = (curYear - df['Yr Sold']) + ((curMonth - df['Mo Sold']) / 12)

    df.drop('Mo Sold', axis=1, inplace=True)  # we don't need it anymore

    print(df.shape)

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

    ordinal_cols1 = [
        "Exter Qual", 'Exter Cond', 'Bsmt Qual', 'Bsmt Cond', 'Bsmt Exposure', 'Heating QC', 'Kitchen Qual',
        'Fireplace Qu',
        'Garage Qual', 'Garage Cond',
    ]
    ordinal_cols2 = [
        'Lot Shape'
    ]
    ordinal_cols3 = [
        'Utilities'
    ]
    ordinal_cols4 = [
        'Land Slope'
    ]
    ordinal_cols5 = [
        'BsmtFin Type 1', 'BsmtFin Type 2'
    ]
    ordinal_cols6 = [
        'Electrical'
    ]
    ordinal_cols7 = [
        'Functional'
    ]
    ordinal_cols8 = [
        'Garage Finish'
    ]
    ordinal_cols9 = [
        'Paved Drive'
    ]
    all_ord_cols = [ordinal_cols1, ordinal_cols2, ordinal_cols3, ordinal_cols4, ordinal_cols5, ordinal_cols6,
                    ordinal_cols7,
                    ordinal_cols8, ordinal_cols9]
    all_ord_elements = ordinal_cols1 + ordinal_cols2 + ordinal_cols3 + ordinal_cols4 + ordinal_cols5 + ordinal_cols6 +ordinal_cols7 + ordinal_cols8 + ordinal_cols9

    mapping1 = {
        'Ex': 5,
        'Gd': 4,
        'TA': 3,
        'Fa': 2,
        'Po': 1,
        'NA': 0,
    }
    mapping2 = {
        'Reg': 4,
        'IR1': 3,
        'IR2': 2,
        'IR3': 1
    }
    mapping3 = {
        'AllPub': 4,
        'NoSewr': 3,
        'NoSeWa': 2,
        'ELO': 1
    }
    mapping4 = {
        'Gtl': 1,
        'Mod': 2,
        'Sev': 3
    }
    mapping5 = {
        'GLQ': 6,
        'ALQ': 5,
        'BLQ': 4,
        'Rec': 3,
        'LwQ': 2,
        'Unf': 1,
        'NA': 0
    }
    mapping6 = {

        'SBrkr': 5,
        'FuseA': 4,
        'FuseF': 3,
        'FuseP': 2,
        'Mix': 1
    }
    mapping7 = {
        'Typ': 8,
        'Min1': 7,
        'Min2': 6,
        'Mod': 5,
        'Maj1': 4,
        'Maj2': 3,
        'Sev': 2,
        'Sal': 1
    }
    mapping8 = {
        'Fin': 3,
        'RFn': 2,
        'Unf': 1,
        'NA': 0
    }
    mapping9 = {
        'Y': 3,
        'P': 2,
        'N': 1,
        'NA': 0
    }

    all_ord_maps = [mapping1, mapping2, mapping3, mapping4, mapping5, mapping6, mapping7, mapping8, mapping9]


    class CustomOrdinalTransformer(BaseEstimator, TransformerMixin):
        def __init__(self, mapping, cols):
            self.mapping = mapping
            self.cols = cols

        def fit(self, X, y=None):
            return self

        def transform(self, X):
            encoder = ce.OrdinalEncoder(mapping=self.mapping, return_df=True, cols=self.cols)
            return pd.DataFrame(encoder.fit_transform(X), columns=X.columns)

    def myFeatureNames(transformer, feature_names):
        return [col for col in feature_names]

    #things are done sequentially inside pipeline
    cont_transformer = Pipeline([
        ('imputer1', SimpleImputer(strategy='mean')),
        ('scale', StandardScaler()),
        ('poly', PolynomialFeatures(degree=2))
    ])

    nom_transformer = Pipeline([
        ('imputer2', SimpleImputer(strategy='most_frequent')),
        ('ohe', OneHotEncoder())
    ])

    ord_transformer = Pipeline([
        ('imputer_ord', SimpleImputer(strategy='most_frequent')),
        ('ord_enc', OrdinalEncoder())
    ])

    full_trans = ColumnTransformerDF([
        ('cont', cont_transformer, cont_cols),
        #('skewness', FunctionTransformer(np.log1p, feature_names_out=myFeatureNames), cont_cols),
        ('nom',nom_transformer , disc_nom_cols),
        ('ord', ord_transformer, all_ord_elements),
        #('model', Lasso(alpha=0.1))


    ],
        remainder='passthrough'
    )
    train, test = train_test_split(df, test_size=0.2, random_state=42)
    x_train, y_train = train[list(train.columns)[:-1]], train[list(train.columns)[-1]]
    x_test, y_test = test[list(test.columns)[:-1]], test[list(test.columns)[-1]]
    '''x_train.drop('Bsmt Full Bath 1', inplace=True, axis=1)
    x_train.drop('Bsmt Half Bath 1', inplace=True, axis=1)
    x_train.drop('Garage Yr Blt 122', inplace=True, axis=1)
    x_train.drop('Garage Cars 1', inplace=True, axis=1)'''
    x_train_trans  = full_trans.fit_transform_df(x_train)
    x_test_trans  = full_trans.fit_transform_df(x_test)
    tr_cols = x_train_trans.columns
    tst_cols = x_test_trans.columns
    for c in tr_cols:
        if c not in tst_cols:
            print(c)
    model = Lasso(alpha=0.1)
    #model.fit(x_train_trans, y_train)
    #preds = model.predict(x_test_trans)
    #print(preds)
    '''for c in list(x_train_trans.columns):
        if x_train_trans[c].isna().sum() > 0:
            print(c,x_train_trans[c].isna().sum())'''

        #if x_test_trans[c].isna().sum() > 0:
            #print(c,x_test_trans[c].isna().sum())