import os
import json
import gzip
import pickle
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    precision_score, balanced_accuracy_score, recall_score, f1_score, confusion_matrix
)
from sklearn.model_selection import GridSearchCV

def preprocess_data(df):
    """Realiza limpieza y transformación de datos."""
    df = df.rename(columns={'default payment next month': 'default'})
    df.drop(columns=['ID'], inplace=True, errors='ignore')
    df.dropna(inplace=True)
    df = df[(df["EDUCATION"] != 0) & (df["MARRIAGE"] != 0)]
    df["EDUCATION"] = df["EDUCATION"].apply(lambda x: 4 if x > 4 else x)
    return df

def create_pipeline(cat_features, num_features=10):
    """Construye un pipeline de preprocesamiento y clasificación."""
    transformer = ColumnTransformer([
        ('cat', OneHotEncoder(handle_unknown='ignore'), cat_features)],
        remainder='passthrough'
    )
    return Pipeline([
        ('preprocessor', transformer),
        ('scaler', MinMaxScaler()),
        ('feature_selection', SelectKBest(score_func=f_classif, k=num_features)),
        ('classifier', LogisticRegression(max_iter=500, random_state=42))
    ])

def tune_hyperparameters(pipeline, x_train, y_train):
    """Optimiza hiperparámetros usando validación cruzada."""
    params = {
        'feature_selection__k': range(1, 11),
        'classifier__C': [0.001, 0.01, 0.1, 1, 10, 100],
        'classifier__penalty': ['l1', 'l2'],
        'classifier__solver': ['liblinear'],
        "classifier__max_iter": [100, 200]
    }
    return GridSearchCV(
        pipeline, param_grid=params, cv=10, scoring='balanced_accuracy', n_jobs=-1, refit=True
    )

def compute_metrics(y_true, y_pred, dataset):
    """Calcula y retorna métricas de evaluación."""
    return {
        'type': 'metrics',
        'dataset': dataset,
        'precision': precision_score(y_true, y_pred),
        'balanced_accuracy': balanced_accuracy_score(y_true, y_pred),
        'recall': recall_score(y_true, y_pred),
        'f1_score': f1_score(y_true, y_pred)
    }

def confusion_matrix_data(y_true, y_pred, dataset):
    """Calcula la matriz de confusión."""
    cm = confusion_matrix(y_true, y_pred)
    return {
        'type': 'cm_matrix',
        'dataset': dataset,
        'true_0': {"predicted_0": int(cm[0, 0]), "predicted_1": int(cm[0, 1])},
        'true_1': {"predicted_0": int(cm[1, 0]), "predicted_1": int(cm[1, 1])}
    }

def save_trained_model(filepath, estimator):
    """Guarda el modelo entrenado en un archivo comprimido."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with gzip.open(filepath, 'wb') as f:
        pickle.dump(estimator, f)

def main():
    """Ejecuta el flujo completo de procesamiento, entrenamiento y evaluación."""
    test_data = pd.read_csv("files/input/test_data.csv.zip", compression="zip")
    train_data = pd.read_csv("files/input/train_data.csv.zip", compression="zip")
    test_data, train_data = map(preprocess_data, [test_data, train_data])
    
    x_train, y_train = train_data.drop(columns=['default']), train_data['default']
    x_test, y_test = test_data.drop(columns=['default']), test_data['default']
    categorical_features = x_train.select_dtypes(include=['object', 'category']).columns.tolist()
    
    pipeline = create_pipeline(categorical_features)
    grid_search = tune_hyperparameters(pipeline, x_train, y_train)
    grid_search.fit(x_train, y_train)
    
    save_trained_model("files/models/model.pkl.gz", grid_search)
    
    metrics = [
        compute_metrics(y_train, grid_search.predict(x_train), 'train'),
        compute_metrics(y_test, grid_search.predict(x_test), 'test'),
        confusion_matrix_data(y_train, grid_search.predict(x_train), 'train'),
        confusion_matrix_data(y_test, grid_search.predict(x_test), 'test')
    ]
    
    os.makedirs("files/output/", exist_ok=True)
    with open("files/output/metrics.json", "w") as f:
        for metric in metrics:
            f.write(json.dumps(metric) + "\n")

if __name__ == "__main__":
    main()
#
