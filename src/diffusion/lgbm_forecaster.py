"""基于 LightGBM 的多期限产业增长预测模型 (LightGBM Multi-Horizon Growth Forecaster).

严格遵循 V1.1 数据真实性要求：
1. 区分四重验证状态：is_algorithm_fitted, is_data_verified, is_out_of_sample_validated, is_calibrated。
2. 准入门禁：正式训练接口 fit_verified 仅接受 quality_status=VERIFIED 的真实样本。
3. 严禁 MANUAL_REFERENCE 或 SYNTHETIC 绕过门禁激活正式模型；提供独立的 fit_experimental 实验模式。
4. 实验模型可拟合收敛，但标记为 EXPERIMENTAL，绝不产生 is_verified=True。
5. 数据不足或未核验时，不输出虚构成长倍数，明确说明准入拒绝原因。
"""

import numpy as np
import pandas as pd
import lightgbm as lgb
from typing import Dict, Any, List, Tuple, Optional
from src.core.types import ModelTrainingStatus, ModelStage, DataQualityStatus, ModelVerificationInfo


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
        # 核心多维状态
        self.training_status: ModelTrainingStatus = ModelTrainingStatus.MODEL_NOT_TRAINED
        self.model_stage: ModelStage = ModelStage.UNTRAINED
        self.is_algorithm_fitted: bool = False
        self.is_data_verified: bool = False
        self.is_out_of_sample_validated: bool = False
        self.is_calibrated: bool = False
        self.is_synthetic: bool = False
        self.sample_count: int = 0
        self.model_version: str = "v1.1-p0-lgbm"
        self.dataset_version: Optional[str] = None
        self.rejection_reason: Optional[str] = None

    def get_verification_info(self) -> ModelVerificationInfo:
        """获取当前模型的多维验证状态详情"""
        return ModelVerificationInfo(
            stage=self.model_stage,
            is_algorithm_fitted=self.is_algorithm_fitted,
            is_data_verified=self.is_data_verified,
            is_out_of_sample_validated=self.is_out_of_sample_validated,
            is_calibrated=self.is_calibrated,
            dataset_version=self.dataset_version,
            model_version=self.model_version,
            rejection_reason=self.rejection_reason
        )

    def fit_verified(
        self,
        features: Optional[Any] = None,
        labels_by_horizon: Optional[Dict[int, Any]] = None,
        quality_status: DataQualityStatus = DataQualityStatus.UNVERIFIED,
        dataset_version: str = "v1.1-verified"
    ) -> Dict[str, Any]:
        """正式训练准入接口：仅接受经过逐项审计与证明的真实 VERIFIED 数据集"""
        self.is_synthetic = False

        # 1. 强制数据质量状态审计门禁 (禁止凭普通数组、MANUAL_REFERENCE、SYNTHETIC 或 UNVERIFIED 启动正式训练)
        self.sample_count = len(features) if features is not None else 0
        if features is None or self.sample_count < self.MIN_REAL_TRAINING_SAMPLES:
            self.training_status = ModelTrainingStatus.MODEL_NOT_TRAINED
            self.model_stage = ModelStage.UNTRAINED
            self.is_data_verified = False
            self.rejection_reason = f"真实验证训练样本不足 ({self.sample_count} < {self.MIN_REAL_TRAINING_SAMPLES})，模型保持未训练状态。"
            return {
                "status": ModelTrainingStatus.MODEL_NOT_TRAINED.value,
                "model_stage": self.model_stage.value,
                "is_verified": False,
                "sample_count": self.sample_count,
                "min_required": self.MIN_REAL_TRAINING_SAMPLES,
                "message": self.rejection_reason
            }

        # 1. 强制数据质量状态审计门禁 (禁止凭普通数组、MANUAL_REFERENCE、SYNTHETIC 或 UNVERIFIED 启动正式训练)
        if quality_status != DataQualityStatus.VERIFIED:
            self.training_status = ModelTrainingStatus.MODEL_NOT_TRAINED
            self.model_stage = ModelStage.UNTRAINED
            self.is_data_verified = False
            self.is_algorithm_fitted = False
            self.rejection_reason = (
                f"正式训练准入门禁拒绝：数据质量状态为 {quality_status.value if hasattr(quality_status, 'value') else quality_status}，"
                f"非 VERIFIED 审计通过状态。MANUAL_REFERENCE 仅允许通过 fit_experimental 运行研究实验模式。"
            )
            return {
                "status": ModelTrainingStatus.MODEL_NOT_TRAINED.value,
                "model_stage": self.model_stage.value,
                "is_verified": False,
                "sample_count": self.sample_count,
                "min_required": self.MIN_REAL_TRAINING_SAMPLES,
                "rejection_reason": self.rejection_reason,
                "message": self.rejection_reason
            }

        # 检查每个期限的标签成熟度与完整性
        for horizon in [1, 3, 5]:
            if horizon not in labels_by_horizon or len(labels_by_horizon[horizon]) != len(features):
                self.training_status = ModelTrainingStatus.MODEL_NOT_TRAINED
                self.model_stage = ModelStage.UNTRAINED
                self.rejection_reason = f"缺少 {horizon} 年期的有效真实标签数据。"
                return {
                    "status": ModelTrainingStatus.MODEL_NOT_TRAINED.value,
                    "model_stage": self.model_stage.value,
                    "is_verified": False,
                    "message": self.rejection_reason
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
        self.model_stage = ModelStage.VERIFIED_PRODUCTION
        self.is_algorithm_fitted = True
        self.is_data_verified = True
        self.is_synthetic = False
        self.sample_count = len(features)
        self.dataset_version = dataset_version
        self.rejection_reason = None
        return {
            "status": ModelTrainingStatus.TRAINED.value,
            "model_stage": self.model_stage.value,
            "is_verified": True,
            "sample_count": self.sample_count,
            "dataset_version": self.dataset_version,
            "message": "模型已使用逐项核验的真实验证数据集完成正式训练。"
        }

    def fit_experimental(
        self,
        features: Any,
        labels_by_horizon: Dict[int, Any],
        dataset_version: str = "v1.1-tech-ref-experimental"
    ) -> Dict[str, Any]:
        """实验性拟合接口：支持基于 MANUAL_REFERENCE 待逐项核验样本进行算法探索。
        
        严格约束：
        - 模型状态标记为 EXPERIMENTAL。
        - is_data_verified 严格为 False。
        - 后续所有预测输出 is_verified 严格为 False，绝不产生正式预测。
        - 仅在对应期限已成熟且非 None 的真实标签样本上拟合。
        """
        if hasattr(features, "values"):
            features_df = features
        else:
            features_df = pd.DataFrame(features, columns=self.feature_names)

        fitted_horizons = []
        for horizon in [1, 3, 5]:
            if horizon not in labels_by_horizon:
                continue

            raw_y = labels_by_horizon[horizon]
            # 过滤未成熟 (None / NaN) 标签
            valid_mask = pd.notna(raw_y) & [y is not None for y in raw_y]
            if np.sum(valid_mask) < 5:
                continue

            valid_X = features_df[valid_mask].values if hasattr(features_df, "values") else np.array(features)[valid_mask]
            valid_y = np.array([float(y) for y in np.array(raw_y)[valid_mask]])

            reg = lgb.LGBMRegressor(
                n_estimators=60,
                learning_rate=0.06,
                num_leaves=12,
                max_depth=4,
                random_state=self.random_state,
                verbose=-1
            )
            reg.fit(valid_X, valid_y)
            self.models[horizon] = reg
            fitted_horizons.append(horizon)

        if not fitted_horizons:
            self.training_status = ModelTrainingStatus.INSUFFICIENT_DATA
            self.model_stage = ModelStage.UNTRAINED
            return {
                "status": ModelTrainingStatus.INSUFFICIENT_DATA.value,
                "model_stage": self.model_stage.value,
                "message": "无有效成熟标签可供实验性拟合。"
            }

        self.training_status = ModelTrainingStatus.EXPERIMENTAL
        self.model_stage = ModelStage.EXPERIMENTAL
        self.is_algorithm_fitted = True
        self.is_data_verified = False
        self.is_synthetic = False
        self.sample_count = len(features_df)
        self.dataset_version = dataset_version
        self.rejection_reason = "模型基于人工参考数据 (MANUAL_REFERENCE) 拟合，仅供实验研究，禁止作为正式预测。"
        return {
            "status": "TRAINED_EXPERIMENTAL_RESEARCH",
            "model_stage": ModelStage.EXPERIMENTAL.value,
            "is_verified": False,
            "fitted_horizons": fitted_horizons,
            "sample_count": self.sample_count,
            "dataset_version": self.dataset_version,
            "warning": self.rejection_reason
        }

    def train(self, features: Optional[np.ndarray] = None, labels_by_horizon: Optional[Dict[int, np.ndarray]] = None):
        """兼容性训练接口 (默认执行严格准入)"""
        return self.fit_verified(features, labels_by_horizon if labels_by_horizon else {}, quality_status=DataQualityStatus.UNVERIFIED)

    def fit_from_historical_tech_samples(self) -> Dict[str, Any]:
        """从历史前沿技术样本中加载并进行实验性研究拟合 (严格标记为 EXPERIMENTAL)"""
        from src.core.database import db_manager
        df_samples = None
        try:
            df_samples = db_manager.query_df("SELECT * FROM tech_training_samples")
        except Exception:
            pass

        if df_samples is None or df_samples.empty:
            from src.ingestion.tech_dataset import HistoricalTechDatasetCollector
            collector = HistoricalTechDatasetCollector()
            records = collector.get_historical_tech_training_samples()
            df_samples = pd.DataFrame(records)

        feature_cols = [
            "velocity_1y", "acceleration", "active_institutions",
            "github_momentum", "macro_multiplier", "entropy_score", "novelty_score"
        ]
        X = df_samples[feature_cols].copy()
        labels_by_horizon = {
            1: df_samples["target_1y"].tolist(),
            3: df_samples["target_3y"].tolist(),
            5: df_samples["target_5y"].tolist()
        }
        # 因为技术时序数据当前为 MANUAL_REFERENCE，必须路由至 fit_experimental
        return self.fit_experimental(
            features=X,
            labels_by_horizon=labels_by_horizon,
            dataset_version="v1.1-historical-tech-dataset"
        )

    def train_on_synthetic_for_testing(self, n_samples: int = 300) -> Dict[str, Any]:
        """【仅供测试与算法演练使用】基于合成模拟特征测试 LightGBM 算法收敛性"""
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

        self.training_status = ModelTrainingStatus.EXPERIMENTAL
        self.model_stage = ModelStage.EXPERIMENTAL
        self.is_algorithm_fitted = True
        self.is_data_verified = False
        self.is_synthetic = True
        self.sample_count = n_samples
        self.dataset_version = "synthetic-test-v1.0"
        return {
            "status": "TRAINED_SYNTHETIC_TESTING_ONLY",
            "model_stage": ModelStage.EXPERIMENTAL.value,
            "is_verified": False,
            "is_synthetic": True,
            "sample_count": n_samples,
            "warning": "模型使用合成模拟数据训练，预测结果不可用于生产决策。"
        }

    def predict_growth(self, features: Dict[str, float]) -> Dict[str, Any]:
        """对单技术主题输入特征预测未来增长倍数。
        
        若模型未拟合，返回 MODEL_NOT_TRAINED 且倍数为 None。
        若模型处于 EXPERIMENTAL 实验阶段，返回 UNVERIFIED_RESEARCH 且 is_verified 严格为 False。
        仅当模型处于 VERIFIED_PRODUCTION 时，才输出正式预测。
        """
        if not self.is_algorithm_fitted or not self.models:
            return {
                "status": ModelTrainingStatus.MODEL_NOT_TRAINED.value,
                "model_stage": ModelStage.UNTRAINED.value,
                "is_verified": False,
                "is_synthetic": False,
                "dataset_version": self.dataset_version,
                "model_version": self.model_version,
                "forecasts": {
                    "1y": {"point_estimate": None, "ci_low": None, "ci_high": None, "scenarios": {}},
                    "3y": {"point_estimate": None, "ci_low": None, "ci_high": None, "scenarios": {}},
                    "5y": {"point_estimate": None, "ci_low": None, "ci_high": None, "scenarios": {}}
                },
                "driver_feature_importance": {},
                "message": "真实样本不足或未达准入门槛，LightGBM模型未训练，不输出虚构预测成长倍数。"
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
            if horizon in self.models:
                pt_est = float(self.models[horizon].predict(feat_vector)[0])
                error_margin = pt_est * (0.12 * horizon)
                ci_low = max(0.2, pt_est - error_margin)
                ci_high = pt_est + error_margin

                bear_scenario = max(0.2, pt_est * (0.75 - 0.05 * horizon))
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
            else:
                predictions[f"{horizon}y"] = {
                    "point_estimate": None,
                    "ci_low": None,
                    "ci_high": None,
                    "scenarios": {}
                }

        # 特征重要性
        driver_importance = {}
        if 3 in self.models:
            imp_3y = self.models[3].feature_importances_
            tot_imp = sum(imp_3y) + 1e-6
            driver_importance = {name: round(float(imp / tot_imp), 3) for name, imp in zip(self.feature_names, imp_3y)}

        is_verified = (self.model_stage == ModelStage.VERIFIED_PRODUCTION and self.is_data_verified and not self.is_synthetic)
        status_str = "PREDICTED" if is_verified else "UNVERIFIED_RESEARCH"

        return {
            "status": status_str,
            "model_stage": self.model_stage.value,
            "is_verified": is_verified,
            "is_synthetic": self.is_synthetic,
            "dataset_version": self.dataset_version,
            "model_version": self.model_version,
            "forecasts": predictions,
            "driver_feature_importance": driver_importance,
            "warning": None if is_verified else "预测基于实验性/未核验模型，仅供研究参考，非真实验证正式预测。"
        }
