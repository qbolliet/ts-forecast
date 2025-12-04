"""Opera adapter for sklearn-compatible ensemble forecasting.

This module provides a wrapper around the opera library's Mixture class
for online expert aggregation and ensemble learning.
"""
import numpy as np
from typing import Any, Dict, Optional
from sklearn.base import BaseEstimator, RegressorMixin


class OperaAdapter(BaseEstimator, RegressorMixin):
    """Sklearn-compatible adapter for opera Mixture ensemble models.

    This adapter wraps the opera library's online expert aggregation algorithms,
    enabling integration with sklearn workflows. Opera performs sequential
    predictions by combining expert forecasts based on their past performance.

    The adapter supports both batch fitting and incremental learning through
    partial_fit(), making it suitable for online learning scenarios.

    Args:
        model: Opera aggregation model name. Options include:
            - "BOA": Bernstein Online Aggregation
            - "MLpol": Polynomial potential aggregation
            - "EWA": Exponentially Weighted Average
            - "FS": Fixed Share
            - "Ridge": Ridge regression
            Defaults to "BOA".
        coefficients: Initial coefficient distribution. Options include:
            - "uniform": Equal weights for all experts
            - "ridge": Ridge-regularized weights
            Defaults to "uniform".
        loss_type: Loss function for evaluating expert performance.
            Options include "mse", "mae", "percentage". Defaults to "mse".
        loss_gradient: If True, uses gradient-based loss. Defaults to True.
        parameters: Additional parameters for the aggregation model as a
            dictionary. Defaults to None.

    Attributes:
        mixture_: The fitted opera Mixture instance.
        is_fitted_: Boolean indicating if the model has been fitted.

    Raises:
        ImportError: If opera package is not installed.

    Examples:
        Basic ensemble of expert forecasts::

            import numpy as np
            import pandas as pd
            from tsforecast.adapters import OperaAdapter

            # Expert predictions (e.g., from different models)
            # Shape: (n_samples, n_experts)
            expert_predictions = np.array([
                [10.2, 9.8, 10.5],   # Sample 1: predictions from 3 experts
                [11.1, 10.9, 11.3],  # Sample 2
                [9.5, 9.7, 9.3],     # Sample 3
            ])

            # True values
            y_true = np.array([10.0, 11.0, 9.6])

            # Create and fit adapter
            adapter = OperaAdapter(model="BOA", loss_type="mse")
            adapter.fit(expert_predictions, y_true)

            # Combine new expert predictions
            new_experts = np.array([[12.1, 11.9, 12.3]])
            ensemble_prediction = adapter.predict(new_experts)

        Online learning with partial_fit::

            # Initial training
            adapter = OperaAdapter(model="EWA")
            adapter.fit(X_initial, y_initial)

            # Incremental updates as new data arrives
            for X_batch, y_batch in data_stream:
                adapter.partial_fit(X_batch, y_batch)
                predictions = adapter.predict(X_next)

        Comparing different aggregation strategies::

            from sklearn.model_selection import cross_val_score

            # Test different opera models
            for model_name in ["BOA", "MLpol", "EWA"]:
                adapter = OperaAdapter(model=model_name)
                scores = cross_val_score(
                    adapter, X_experts, y_true,
                    cv=5, scoring='neg_mean_squared_error'
                )
                print(f"{model_name} MSE: {-scores.mean():.3f}")

        Using with GridSearchCV::

            from sklearn.model_selection import GridSearchCV

            param_grid = {
                'model': ['BOA', 'EWA', 'MLpol'],
                'loss_type': ['mse', 'mae'],
            }

            grid_search = GridSearchCV(
                OperaAdapter(),
                param_grid,
                cv=5,
                scoring='neg_mean_squared_error'
            )
            grid_search.fit(X_experts, y_true)

            print(f"Best model: {grid_search.best_params_}")
    """

    def __init__(
        self,
        model: str = "BOA",
        coefficients: str = "uniform",
        loss_type: str = "mse",
        loss_gradient: bool = True,
        parameters: Optional[Dict[str, Any]] = None
    ) -> None:
        """Initialize the OperaAdapter.

        Args:
            model: Opera aggregation model name.
            coefficients: Initial coefficient distribution.
            loss_type: Loss function for expert evaluation.
            loss_gradient: Whether to use gradient-based loss.
            parameters: Additional model parameters.

        Examples:
            Creating adapters with different configurations::

                from tsforecast.adapters import OperaAdapter

                # Bernstein Online Aggregation with MSE loss
                adapter1 = OperaAdapter(model="BOA", loss_type="mse")

                # Exponentially Weighted Average with MAE loss
                adapter2 = OperaAdapter(model="EWA", loss_type="mae")

                # With custom parameters
                adapter3 = OperaAdapter(
                    model="Ridge",
                    parameters={"lambda": 0.1}
                )
        """
        self.model = model
        self.coefficients = coefficients
        self.loss_type = loss_type
        self.loss_gradient = loss_gradient
        self.parameters = parameters
        self.mixture_ = None
        self.is_fitted_ = False

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray
    ) -> "OperaAdapter":
        """Fit the opera ensemble model on expert predictions.

        Args:
            X: Expert predictions with shape (n_samples, n_experts).
                Each column represents predictions from one expert model.
            y: True target values with shape (n_samples,).

        Returns:
            self: The fitted adapter instance.

        Raises:
            ImportError: If opera package is not installed.
            ValueError: If X and y have incompatible shapes.

        Examples:
            Training the ensemble::

                import numpy as np
                from tsforecast.adapters import OperaAdapter

                # Three experts, 100 samples
                X_experts = np.random.randn(100, 3)
                y_true = np.random.randn(100)

                adapter = OperaAdapter(model="BOA")
                adapter.fit(X_experts, y_true)

                # Check fitted mixture
                print(adapter.mixture_.coefficients)  # Expert weights

            Fitting with expert predictions from real models::

                from sklearn.linear_model import Ridge, Lasso
                from sklearn.ensemble import RandomForestRegressor

                # Train multiple models (experts)
                model1 = Ridge().fit(X_train, y_train)
                model2 = Lasso().fit(X_train, y_train)
                model3 = RandomForestRegressor().fit(X_train, y_train)

                # Get predictions on validation set
                expert_preds = np.column_stack([
                    model1.predict(X_val),
                    model2.predict(X_val),
                    model3.predict(X_val)
                ])

                # Fit opera to combine experts
                adapter = OperaAdapter()
                adapter.fit(expert_preds, y_val)
        """
        # Lazy import of opera
        try:
            from opera import Mixture
        except ImportError as e:
            raise ImportError(
                "OperaAdapter requires the 'opera' package. "
                "Install it with: pip install ts-forecast[adapters-opera] "
                "or: pip install opera"
            ) from e

        # Initialisation du modèle Mixture
        # Note: L'entraînement initial est effectué lors de l'initialisation
        self.mixture_ = Mixture(
            y=y,
            experts=X,
            model=self.model,
            coefficients=self.coefficients,
            loss_type=self.loss_type,
            loss_gradient=self.loss_gradient,
            parameters=self.parameters
        )

        self.is_fitted_ = True
        return self

    def partial_fit(
        self,
        X: np.ndarray,
        y: np.ndarray
    ) -> "OperaAdapter":
        """Incrementally update the ensemble model with new data.

        This method enables online learning by updating expert weights
        based on new observations without refitting from scratch.

        Args:
            X: New expert predictions with shape (n_samples, n_experts).
            y: New true target values with shape (n_samples,).

        Returns:
            self: The updated adapter instance.

        Raises:
            ImportError: If opera package is not installed.

        Examples:
            Online learning scenario::

                from tsforecast.adapters import OperaAdapter
                import numpy as np

                adapter = OperaAdapter()

                # Initial fit
                adapter.fit(X_initial, y_initial)

                # Stream of new data
                for X_new, y_new in data_batches:
                    # Update model with new observations
                    adapter.partial_fit(X_new, y_new)

                    # Model adapts expert weights based on performance
                    current_weights = adapter.mixture_.coefficients

            Incremental learning from scratch::

                adapter = OperaAdapter()

                # partial_fit automatically calls fit if not fitted
                for X_batch, y_batch in data_stream:
                    adapter.partial_fit(X_batch, y_batch)
        """
        # Si le modèle n'est pas encore entraîné, appeler fit
        if not self.is_fitted_:
            return self.fit(X, y)

        # Lazy import
        try:
            from opera import Mixture
        except ImportError as e:
            raise ImportError(
                "OperaAdapter requires the 'opera' package. "
                "Install it with: pip install ts-forecast[adapters-opera] "
                "or: pip install opera"
            ) from e

        # Mise à jour incrémentale du modèle
        self.mixture_.update(new_experts=X, new_y=y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Generate ensemble predictions from expert forecasts.

        Combines expert predictions using the learned weights without
        updating the model.

        Args:
            X: Expert predictions with shape (n_samples, n_experts).

        Returns:
            Ensemble predictions with shape (n_samples,).

        Raises:
            ValueError: If model is not fitted.

        Examples:
            Making predictions::

                # Fit on historical data
                adapter.fit(X_train_experts, y_train)

                # Combine expert predictions for new data
                y_pred = adapter.predict(X_test_experts)

                # Predictions are weighted combination of experts
                print(y_pred.shape)  # (n_test_samples,)

            Real-world prediction workflow::

                from tsforecast.adapters import OperaAdapter
                import numpy as np

                # Train experts and get their predictions on test set
                test_expert_preds = np.column_stack([
                    model1.predict(X_test),
                    model2.predict(X_test),
                    model3.predict(X_test)
                ])

                # Combine using trained opera adapter
                ensemble_pred = adapter.predict(test_expert_preds)
        """
        if not self.is_fitted_:
            raise ValueError(
                "This OperaAdapter instance is not fitted yet. "
                "Call 'fit' or 'partial_fit' before using 'predict'."
            )

        # Prédiction sans mise à jour des coefficients
        return self.mixture_.predict(new_experts=X)

    def score(
        self,
        X: np.ndarray,
        y: np.ndarray,
        sample_weight: Optional[np.ndarray] = None
    ) -> float:
        """Calculate R² score for ensemble predictions.

        This method is required for GridSearchCV compatibility.

        Args:
            X: Expert predictions for test data.
            y: True target values.
            sample_weight: Sample weights for scoring. Defaults to None.

        Returns:
            R² score (coefficient of determination).

        Examples:
            Evaluating ensemble performance::

                from tsforecast.adapters import OperaAdapter

                adapter.fit(X_train, y_train)
                r2 = adapter.score(X_test, y_test)
                print(f"R² score: {r2:.3f}")

            Comparing with individual experts::

                from sklearn.metrics import r2_score

                # Ensemble performance
                ensemble_r2 = adapter.score(X_test_experts, y_test)

                # Individual expert performance
                for i in range(X_test_experts.shape[1]):
                    expert_r2 = r2_score(y_test, X_test_experts[:, i])
                    print(f"Expert {i+1} R²: {expert_r2:.3f}")

                print(f"Ensemble R²: {ensemble_r2:.3f}")
        """
        from sklearn.metrics import r2_score

        y_pred = self.predict(X)
        return r2_score(y, y_pred, sample_weight=sample_weight)
