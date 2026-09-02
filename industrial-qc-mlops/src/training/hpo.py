"""Hyperparameter optimization with Optuna for YOLOv8."""

import os
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import mlflow
import optuna
from optuna.integration import MLflowCallback
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler

from src.config import Settings, get_settings
from src.logger import get_logger
from src.training.train import YOLOv8Trainer

logger = get_logger(__name__)


class HyperparameterOptimizer:
    """
    Hyperparameter optimization for YOLOv8 using Optuna.

    Optimizes:
    - Learning rate (lr0)
    - Weight decay
    - Momentum
    - Image size
    - Batch size
    - Augmentation parameters
    """

    def __init__(
        self,
        settings: Optional[Settings] = None,
        data_yaml: Optional[Path] = None,
        study_name: Optional[str] = None,
        storage: Optional[str] = None,
    ) -> None:
        """
        Initialize HPO.

        Args:
            settings: Application settings.
            data_yaml: Path to dataset YAML.
            study_name: Optuna study name.
            storage: Optuna storage URL (for distributed optimization).
        """
        self.settings = settings or get_settings()
        self.data_yaml = data_yaml or Path("configs/data.yaml")
        self.study_name = study_name or self.settings.hpo_study_name if hasattr(self.settings, 'hpo_study_name') else "yolov8-hpo"
        self.storage = storage

        # Create sampler and pruner
        sampler = TPESampler(seed=42)
        pruner = MedianPruner(n_startup_trials=5, n_warmup_steps=10)

        # Create study
        direction = "maximize"  # Maximize mAP50-95
        self.study = optuna.create_study(
            study_name=self.study_name,
            storage=self.storage,
            direction=direction,
            sampler=sampler,
            pruner=pruner,
            load_if_exists=True,
        )

        logger.info(
            "HyperparameterOptimizer initialized",
            study_name=self.study_name,
            direction=direction,
        )

    def suggest_hyperparameters(self, trial: optuna.Trial) -> Dict[str, Any]:
        """
        Suggest hyperparameters for a trial.

        Args:
            trial: Optuna trial object.

        Returns:
            Dictionary of suggested hyperparameters.
        """
        return {
            # Optimizer hyperparameters
            "lr0": trial.suggest_float("lr0", 1e-4, 1e-1, log=True),
            "lrf": trial.suggest_float("lrf", 1e-3, 1e-1, log=True),
            "momentum": trial.suggest_float("momentum", 0.8, 0.98),
            "weight_decay": trial.suggest_float("weight_decay", 1e-5, 1e-2, log=True),

            # Model hyperparameters
            "imgsz": trial.suggest_categorical("imgsz", [416, 512, 640]),
            "batch_size": trial.suggest_categorical("batch_size", [8, 16, 32]),

            # Augmentation hyperparameters
            "hsv_h": trial.suggest_float("hsv_h", 0.0, 0.03),
            "hsv_s": trial.suggest_float("hsv_s", 0.0, 0.9),
            "hsv_v": trial.suggest_float("hsv_v", 0.0, 0.6),
            "flipud": trial.suggest_float("flipud", 0.0, 0.1),
            "fliplr": trial.suggest_float("fliplr", 0.3, 0.8),
            "mosaic": trial.suggest_float("mosaic", 0.5, 1.0),
            "mixup": trial.suggest_float("mixup", 0.0, 0.2),
        }

    def objective(self, trial: optuna.Trial) -> float:
        """
        Objective function for Optuna.

        Args:
            trial: Optuna trial object.

        Returns:
            Validation metric to optimize (mAP50-95).
        """
        # Get hyperparameters
        params = self.suggest_hyperparameters(trial)

        # Generate unique run name for this trial
        run_name = f"hpo-trial-{trial.number}"

        try:
            # Create trainer
            trainer = YOLOv8Trainer(settings=self.settings)

            # Override config with trial parameters
            trainer.config.update(params)

            # Train model
            result = trainer.train(
                data_yaml=self.data_yaml,
                epochs=30,  # Reduced epochs for faster HPO
                batch_size=params["batch_size"],
                imgsz=params["imgsz"],
                run_name=run_name,
            )

            if not result["success"]:
                raise RuntimeError(f"Training failed: {result.get('error', 'Unknown error')}")

            # Return metric to optimize
            mAP50_95 = result["metrics"].get("mAP50-95", 0.0)

            # Log trial info
            trial.set_user_attr("mAP50", result["metrics"].get("mAP50", 0.0))
            trial.set_user_attr("precision", result["metrics"].get("precision", 0.0))
            trial.set_user_attr("recall", result["metrics"].get("recall", 0.0))

            logger.info(
                f"Trial {trial.number} completed",
                mAP50_95=mAP50_95,
                params=params,
            )

            return mAP50_95

        except Exception as e:
            logger.error(
                f"Trial {trial.number} failed",
                error=str(e),
                exc_info=True,
            )
            # Return worst value to prune this trial
            raise optuna.TrialPruned()

    def optimize(
        self,
        n_trials: int = 50,
        timeout: Optional[int] = None,
        n_jobs: int = 1,
        show_progress_bar: bool = True,
    ) -> optuna.Study:
        """
        Run hyperparameter optimization.

        Args:
            n_trials: Number of trials to run.
            timeout: Timeout in seconds.
            n_jobs: Number of parallel jobs (-1 for all CPUs).
            show_progress_bar: Show progress bar.

        Returns:
            Optuna study object with results.
        """
        logger.info(
            "Starting hyperparameter optimization",
            n_trials=n_trials,
            timeout=timeout,
            n_jobs=n_jobs,
        )

        # Setup MLflow callback
        mlflow_tracking_uri = self.settings.mlflow_tracking_uri
        mlflow.set_tracking_uri(mlflow_tracking_uri)

        mlflow_callback = MLflowCallback(
            tracking_uri=mlflow_tracking_uri,
            metric_name="mAP50-95",
        )

        try:
            # Run optimization
            self.study.optimize(
                self.objective,
                n_trials=n_trials,
                timeout=timeout,
                n_jobs=n_jobs,
                callbacks=[mlflow_callback.track_in_mlflow()],
                show_progress_bar=show_progress_bar,
            )

            # Log best results
            best_trial = self.study.best_trial
            best_params = self.study.best_params
            best_value = self.study.best_value

            logger.info(
                "Optimization completed",
                best_mAP50_95=best_value,
                best_params=best_params,
                num_trials=len(self.study.trials),
            )

            return self.study

        except KeyboardInterrupt:
            logger.warning("Optimization interrupted by user")
            return self.study

    def get_best_params(self) -> Dict[str, Any]:
        """
        Get best hyperparameters.

        Returns:
            Dictionary of best hyperparameters.
        """
        return self.study.best_params

    def save_results(self, output_path: Path) -> None:
        """
        Save optimization results to file.

        Args:
            output_path: Path to save results.
        """
        import json
        import pandas as pd

        # Best params
        best_params = self.study.best_params
        best_value = self.study.best_value

        # Trial history
        trials_df = self.study.trials_dataframe()

        # Save best params
        results = {
            "best_mAP50-95": best_value,
            "best_params": best_params,
            "num_trials": len(self.study.trials),
        }

        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)

        # Save trials dataframe
        csv_path = output_path.with_suffix(".csv")
        trials_df.to_csv(csv_path)

        logger.info(
            "Results saved",
            json_path=str(output_path),
            csv_path=str(csv_path),
        )

    def plot_results(
        self,
        output_dir: Path,
    ) -> List[Path]:
        """
        Generate visualization plots.

        Args:
            output_dir: Directory to save plots.

        Returns:
            List of paths to generated plots.
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        plot_paths = []

        try:
            # Plot optimization history
            fig1 = optuna.visualization.plot_optimization_history(self.study)
            path1 = output_dir / "optimization_history.png"
            fig1.write_image(str(path1))
            plot_paths.append(path1)

            # Plot parameter importance
            fig2 = optuna.visualization.plot_param_importances(self.study)
            path2 = output_dir / "param_importance.png"
            fig2.write_image(str(path2))
            plot_paths.append(path2)

            # Plot slice
            fig3 = optuna.visualization.plot_slice(self.study)
            path3 = output_dir / "slice.png"
            fig3.write_image(str(path3))
            plot_paths.append(path3)

            # Plot contour
            fig4 = optuna.visualization.plot_contour(self.study)
            path4 = output_dir / "contour.png"
            fig4.write_image(str(path4))
            plot_paths.append(path4)

            logger.info(
                "Plots generated",
                num_plots=len(plot_paths),
            )

        except Exception as e:
            logger.error(
                "Failed to generate plots",
                error=str(e),
            )

        return plot_paths


def run_hpo(
    data_yaml: Path,
    n_trials: int = 50,
    timeout: Optional[int] = 3600,
    output_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    """
    Convenience function to run HPO.

    Args:
        data_yaml: Path to dataset YAML.
        n_trials: Number of trials.
        timeout: Timeout in seconds.
        output_dir: Directory to save results.

    Returns:
        Dictionary with best parameters and metrics.
    """
    settings = get_settings()
    output_dir = output_dir or Path("models/hpo")

    optimizer = HyperparameterOptimizer(
        settings=settings,
        data_yaml=data_yaml,
    )

    # Run optimization
    study = optimizer.optimize(n_trials=n_trials, timeout=timeout)

    # Save results
    results_path = output_dir / "hpo_results.json"
    optimizer.save_results(results_path)

    # Generate plots
    optimizer.plot_results(output_dir / "plots")

    return {
        "best_params": optimizer.get_best_params(),
        "best_value": study.best_value,
        "results_path": str(results_path),
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run HPO for YOLOv8")
    parser.add_argument("--data", type=str, default="configs/data.yaml")
    parser.add_argument("--trials", type=int, default=50)
    parser.add_argument("--timeout", type=int, default=3600)
    parser.add_argument("--output", type=str, default="models/hpo")

    args = parser.parse_args()

    results = run_hpo(
        data_yaml=Path(args.data),
        n_trials=args.trials,
        timeout=args.timeout,
        output_dir=Path(args.output),
    )

    print(f"\nBest mAP50-95: {results['best_value']:.4f}")
    print(f"Best params: {results['best_params']}")
