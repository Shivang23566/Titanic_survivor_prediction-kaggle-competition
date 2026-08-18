import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder

df = pd.read_csv("d:/Kaggle_work/Titanic Survivor prediction competition/train.csv")
split = StratifiedShuffleSplit(n_splits=1, test_size=0.2)
for train_indices, _ in split.split(df, df[["Survived", "Pclass", "Sex"]]):
    X = df.loc[train_indices].copy()

class AgeImputer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        imputer = SimpleImputer(strategy="mean")
        X["Age"] = imputer.fit_transform(X[["Age"]])
        return X

class FeatureEncoder(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        embarked_encoder = OneHotEncoder(handle_unknown="ignore")
        embarked_matrix = embarked_encoder.fit_transform(X[["Embarked"]]).toarray()
        for i, column_name in enumerate(["C", "S", "Q", "N"]):
            X[column_name] = embarked_matrix[:, i]

        sex_encoder = OneHotEncoder(handle_unknown="ignore")
        sex_matrix = sex_encoder.fit_transform(X[["Sex"]]).toarray()
        for i, column_name in enumerate(["Female", "Male"]):
            X[column_name] = sex_matrix[:, i]
        return X

X = AgeImputer().fit_transform(X)
X = FeatureEncoder().fit_transform(X)
print("Female" in X.columns, "Male" in X.columns)
print(X[["Female", "Male"]].head().to_dict())
