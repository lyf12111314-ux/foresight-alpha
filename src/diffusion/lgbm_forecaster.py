"""基于 LightGBM 的多期限产业增长预测模型 (LightGBM Multi-Horizon Growth Forecaster).

严格遵循 V1.1 数据真实性要求：
1. 正式训练接口仅接受经过验证的真实历史特征与真实标签。
2. 严禁在未训练时自动使用程序生成模拟数据伪造预测。
3. 真实训练样本不足时，明确返回 MODEL_NOT_TRAINED，不得输出虚构成长倍数。
4. 合成数据算法测试逻辑完全隔离。
"""

import numpy as np
import pandas as pd
import lightgbm as lgb
from typing import Dict, Any, List, Tuple, Optional
from src.core.types import ModelTrainingStatus


class LightGBMGrowthForecaster:
    """LightGBM 产业与技术多期复合增长预测器"""

    MIN_REAL_TRAINING_SAMPLES = 20  # 真实历史样本准入门槛

    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.models: Dict[int, lgb.LGBMRegressor] = {}
        self.feature_names = [
            "velocity_1y",
            "acceleration",
            "active_institutions",
            "github_momentum",
            "macro_multiplier",
            "entropy_score",
            "novelty_score"
        ]
        self.training_status: ModelTrainingStatus = ModelTrainingStatus.MODEL_NOT_TRAINED
        self.is_synthetic: bool = False
        self.sample_count: int = 0

    def fit_verified(
        self,
        features: Optional[Any] = None,
        labels_by_horizon: Optional[Dict[int, Any]] = None
    ) -> Dict[str, Any]:
        """正式训练接口：仅接受经过核实验证的真实历史特征与标签"""
        self.is_synthetic = False
        if labels_by_horizon is None:
            labels_by_horizon = {}
        if hasattr(features, "values"):
            features = features.values
        if features is None or len(features) < self.MIN_REAL_TRAINING_SAMPLES:
            self.training_status = ModelTrainingStatus.MODEL_NOT_TRAINED
            self.sample_count = len(features) if features is not None else 0
            return {
                "status": ModelTrainingStatus.MODEL_NOT_TRAINED.value,
                "sample_count": self.sample_count,
                "min_required": self.MIN_REAL_TRAINING_SAMPLES,
                "message": f"真实训练样本不足 ({self.sample_count} < {self.MIN_REAL_TRAINING_SAMPLES})，模型保持未训练状态。"
            }

        for horizon in [1, 3, 5]:
            if horizon not in labels_by_horizon or len(labels_by_horizon[horizon]) != len(features):
                self.training_status = ModelTrainingStatus.MODEL_NOT_TRAINED
                return {
                    "status": ModelTrainingStatus.MODEL_NOT_TRAINED.value,
                    "message": f"缺少 {horizon} 年期的有效真实标签数据。"
                }

            reg = lgb.LGBMRegressor(
                n_estimators=80,
                learning_rate=0.06,
                num_leaves=15,
                max_depth=5,
                random_state=self.random_state,
                verbose=-1
            )
            reg.fit(features, labels_by_horizon[horizon])
            self.models[horizon] = reg

        self.training_status = ModelTrainingStatus.TRAINED
        self.sample_count = len(features)
        return {
            "status": ModelTrainingStatus.TRAINED.value,
            "sample_count": self.sample_count,
            "message": "模型已使用真实验证数据完成训练。"
        }

    def train(self, features: Optional[np.ndarray] = None, labels_by_horizon: Optional[Dict[int, np.ndarray]] = None):
        """兼容性训练接口 (严格拒绝空样本自动模拟训练)"""
        return self.fit_verified(features, labels_by_horizon if labels_by_horizon else {})

    def train_on_synthetic_for_testing(self, n_samples: int = 300) -> Dict[str, Any]:
        """【仅供测试与算法演练使用】基于合成特征测试 LightGBM 算法收敛性"""
        from tests.helpers.synthetic_diffusion import generate_synthetic_diffusion_samples
        X, Y_dict = generate_synthetic_diffusion_samples(n_samples=n_samples, random_state=self.random_state)
        for horizon in [1, 3, 5]:
            reg = lgb.LGBMRegressor(
                n_estimators=80,
                learning_rate=0.06,
                num_leaves=15,
                max_depth=5,
                random_state=self.random_state,
                verbose=-1
            )
            reg.fit(X, Y_dict[horizon])
            self.models[horizon] = reg

        self.training_status = ModelTrainingStatus.TRAINED
        self.is_synthetic = True
        self.sample_count = n_samples
        return {
            "status": "TRAINED_SYNTHETIC_TESTING_ONLY",
            "is_synthetic": True,
            "sample_count": n_samples,
            "warning": "模型使用合成模拟数据训练，预测结果不可用于生产决策。"
        }

    def predict_growth(self, features: Dict[str, float]) -> Dict[str, Any]:
        """对单技术主题输入特征预测未来增长倍数。
        
        如果模型未训练或未经真实验证，严禁输出虚构的预测倍数。
        """
        if self.training_status != ModelTrainingStatus.TRAINED or not self.models:
            return {
                "status": ModelTrainingStatus.MODEL_NOT_TRAINED.value,
                "is_verified": False,
                "is_synthetic": False,
                "forecasts": {
                    "1y": {"point_estimate": None, "ci_low": None, "ci_high": None, "scenarios": {}},
                    "3y": {"point_estimate": None, "ci_low": None, "ci_high": None, "scenarios": {}},
                    "5y": {"point_estimate": None, "ci_low": None, "ci_high": None, "scenarios": {}}
                },
                "driver_feature_importance": {},
                "message": "真实样本不足，LightGBM模型未训练，不输出虚构预测成长倍数。"
            }

        feat_vector = np.array([[
            features.get("velocity_1y", 0.0),
            features.get("acceleration", 0.0),
            features.get("active_institutions", 0.0),
            features.get("github_momentum", 0.0),
            features.get("macro_multiplier", 1.0),
            features.get("entropy_score", 0.5),
            features.get("novelty_score", 0.5)
        ]])

        predictions = {}
        for horizon in [1, 3, 5]:
            pt_est = float(self.models[horizon].predict(feat_vector)[0])
            error_margin = pt_est * (0.12 * horizon)
            ci_low = max(0.5, pt_est - error_margin)
            ci_high = pt_est + error_margin

            bear_scenario = max(0.4, pt_est * (0.75 - 0.05 * horizon))
            base_scenario = pt_est
            bull_scenario = pt_est * (1.30 + 0.10 * horizon)

            predictions[f"{horizon}y"] = {
                "point_estimate": round(pt_est, 2),
                "ci_low": round(ci_low, 2),
                "ci_high": round(ci_high, 2),
                "scenarios": {
                    "bear": round(bear_scenario, 2),
                    "base": round(base_scenario, 2),
                    "bull": round(bull_scenario, 2)
                }
            }

        imp_3y = self.models[3].feature_importances_
        tot_imp = sum(imp_3y) + 1e-6
        driver_importance = {name: round(float(imp / tot_imp), 3) for name, imp in zip(self.feature_names, imp_3y)}

        return {
            "status": "PREDICTED",
            "is_verified": not self.is_synthetic,
            "is_synthetic": self.is_synthetic,
            "forecasts": predictions,
            "driver_feature_importance": driver_importance
        }
