import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer

def prepare_data_for_prediction(X, categorical_features, numerical_features, model):

    # Remove any rows with missing values in the target variable, if present
    X = X.dropna().reset_index(drop=True)

    # Remove any columns with a high proportion of missing values
    col_thresh = 0.8  # Proportion of missing values above which a column is dropped
    X = X.dropna(thresh=int(col_thresh*X.shape[0]), axis=1)

    # Remove any rows with missing values in features
    row_thresh = 0.2  # Proportion of missing values above which a row is dropped
    X = X.dropna(thresh=int((1-row_thresh)*X.shape[1]), axis=0)

    
    # Define the column transformer to apply the specified preprocessing steps to the relevant features
    transformer = ColumnTransformer(transformers=[
        ('imputer_categorical', SimpleImputer(strategy='most_frequent'), categorical_features),
        ('imputer_numerical', SimpleImputer(strategy='mean'), numerical_features),
        ('scaler', StandardScaler(), numerical_features),
        ('encoder', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ])

    # Fit the transformer to the input data X and transform it
    X_transformed = transformer.fit_transform(X)

    # Convert the transformed data back into a Pandas DataFrame
    columns = [f'num_{i}' for i in range(len(numerical_features))] + list(transformer.named_transformers_['encoder'].get_feature_names(categorical_features))
    X_transformed = pd.DataFrame(X_transformed, columns=columns)

    # Make predictions using the specified model
    y_pred = model.predict(X_transformed)

    return y_pred
