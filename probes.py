import pandas as pd  # type: ignore
import torch as t
from beartype import beartype as typed
from jaxtyping import Bool, Float
from sklearn.linear_model import LogisticRegression, Ridge  # type: ignore
from sklearn.metrics import r2_score, roc_auc_score  # type: ignore
from sklearn.model_selection import train_test_split  # type: ignore
from torch import Tensor as TT


@typed
def linear_regression_probe(
    embeddings: Float[TT, "n d"], features: Float[TT, "n"]
) -> float:

    X_train, X_test, y_train, y_test = train_test_split(
        embeddings.numpy(), features.numpy(), test_size=0.2
    )
    model = Ridge().fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return r2_score(y_test, y_pred)


@typed
def linear_classification_probe(
    embeddings: Float[TT, "n d"], features: Bool[TT, "n"]
) -> float:

    X_train, X_test, y_train, y_test = train_test_split(
        embeddings.numpy(),
        features.numpy(),
        test_size=0.2,
        random_state=42,
    )
    model = LogisticRegression(max_iter=1000).fit(X_train, y_train)
    y_pred = model.predict_proba(X_test)[:, 1]
    return roc_auc_score(y_test, y_pred)


@typed
def mixed_probe(
    embeddings: Float[TT, "n d"], features: pd.DataFrame
) -> dict[str, float]:
    result = {}
    for column in features.columns:
        dtype = str(features[column].dtype)
        if "float" in dtype:
            y = t.tensor(features[column])
            result[column] = linear_regression_probe(embeddings, y)
        elif "bool" in dtype:
            y = t.tensor(features[column])
            result[column] = linear_classification_probe(embeddings, y)
    return result
