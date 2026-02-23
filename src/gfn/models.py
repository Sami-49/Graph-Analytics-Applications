"""Machine learning models: Logistic Regression, Random Forest, Gradient Boosting."""
from typing import Callable
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
try:
    import xgboost as xgb
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False
    from sklearn.ensemble import HistGradientBoostingClassifier


class ModelBuilder:
    """Factory for baseline ML classifiers."""

    @staticmethod
    def build_logistic_regression(config) -> LogisticRegression:
        """Build Logistic Regression classifier."""
        params = config.models.logistic_regression
        return LogisticRegression(
            C=params.get("C", 1.0),
            max_iter=params.get("max_iter", 1000),
            random_state=config.evaluation.random_state,
            class_weight="balanced",
        )

    @staticmethod
    def build_random_forest(config) -> RandomForestClassifier:
        """Build Random Forest classifier."""
        params = config.models.random_forest
        return RandomForestClassifier(
            n_estimators=params.get("n_estimators", 100),
            max_depth=params.get("max_depth", 10),
            min_samples_split=params.get("min_samples_split", 5),
            random_state=params.get("random_state", 42),
            class_weight="balanced",
            n_jobs=-1,
        )

    @staticmethod
    def build_gradient_boosting(config) -> Callable:
        """Build Gradient Boosting classifier (XGBoost or fallback)."""
        params = config.models.gradient_boosting

        if HAS_XGBOOST:
            kwargs = dict(
                n_estimators=params.get("n_estimators", 100),
                max_depth=params.get("max_depth", 5),
                learning_rate=params.get("learning_rate", 0.1),
                random_state=params.get("random_state", 42),
            )
            try:
                return xgb.XGBClassifier(**kwargs, eval_metric="logloss")
            except TypeError:
                return xgb.XGBClassifier(**kwargs)
        else:
            return HistGradientBoostingClassifier(
                max_iter=params.get("n_estimators", 100),
                max_depth=params.get("max_depth", 5),
                learning_rate=params.get("learning_rate", 0.1),
                random_state=params.get("random_state", 42),
            )

    @staticmethod
    def get_all_models(config) -> dict:
        """Get all three baseline models."""
        return {
            "logistic_regression": ModelBuilder.build_logistic_regression(config),
            "random_forest": ModelBuilder.build_random_forest(config),
            "gradient_boosting": ModelBuilder.build_gradient_boosting(config),
        }
