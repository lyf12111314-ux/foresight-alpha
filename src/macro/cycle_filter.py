"""宏观长中短周期分解与滤波引擎 (Multi-Cycle Decomposition Engine).

集成：
1. 康波长周期 (Kondratiev Wave, ~50年) 结构性技术革命定位
2. 朱格拉设备投资中周期 (Juglar Cycle, 7-10年)
3. 基钦库存短周期 (Kitchin Cycle, ~40个月)
4. Hodrick-Prescott (HP) 滤波趋势与周期分离
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple
from statsmodels.tsa.filters.hp_filter import hpfilter


class MacroCycleFilter:
    """宏观经济多周期分解器"""

    def __init__(self, hp_lambda: float = 1600.0):
        # 季度数据标准 lambda 常用 1600.0
        self.hp_lambda = hp_lambda

    def decompose_hp(self, series: pd.Series) -> Tuple[pd.Series, pd.Series]:
        """使用 HP 滤波将时间序列分解为趋势项 (Trend) 与周期波动项 (Cycle)"""
        clean_s = series.ffill().bfill().dropna()
        if len(clean_s) < 8:
            return clean_s, pd.Series(0.0, index=clean_s.index)
        cycle, trend = hpfilter(clean_s, lamb=self.hp_lambda)
        return trend, cycle

    def analyze_kondratiev_wave(self) -> Dict[str, Any]:
        """康波长周期定性与结构研判框架 (第6次康波浪潮)"""
        return {
            "current_wave": "第六次康波长波 (The 6th Kondratiev Wave)",
            "primary_drivers": ["通用人工智能 (AGI) 与具身智能", "新型电力系统与能源互联网", "合成生物与基因工程"],
            "current_phase": "扩散与繁荣构建期 (Diffusion & Deployment Phase)",
            "phase_description": "底层基础科学突破已初步完成，正由实验室萌芽转向规模化产业落地与生产力赋能阶段。",
            "historical_parallels": "类似于1995-1998年互联网基础设施扩散期，资本向关键硬件、算力管道与核心零部件集中沉淀。",
            "structural_tailwinds": "全社会要素生产率提升压力大，政策倾斜与产业资本高度支持硬科技突破与高端制造升级。"
        }

    def analyze_inventory_cycle(self, macro_df: pd.DataFrame) -> Dict[str, Any]:
        """基钦库存周期状态判别 (主动去库 / 被动去库 / 主动补库 / 被动补库)"""
        pmi_df = macro_df[macro_df["indicator_code"] == "CN_PMI"].sort_values("as_of_date")
        ppi_df = macro_df[macro_df["indicator_code"] == "CN_PPI_YOY"].sort_values("as_of_date")

        if pmi_df.empty or ppi_df.empty:
            return {
                "inventory_phase": "数据不足",
                "phase_code": "UNKNOWN",
                "recommendation": "保持中性观察"
            }

        latest_pmi = float(pmi_df.iloc[-1]["value"])
        latest_ppi = float(ppi_df.iloc[-1]["value"])
        prev_pmi = float(pmi_df.iloc[-2]["value"]) if len(pmi_df) > 1 else latest_pmi
        prev_ppi = float(ppi_df.iloc[-2]["value"]) if len(ppi_df) > 1 else latest_ppi

        pmi_direction = latest_pmi - prev_pmi
        ppi_direction = latest_ppi - prev_ppi

        # 四象限划分
        if pmi_direction >= 0 and ppi_direction >= 0:
            phase = "主动补库存 (Active Restocking)"
            code = "EXPANDING"
            desc = "需求与价格同步回暖，企业积极补充原材料与零部件库存，制造业上游与中游业绩共振向上。"
        elif pmi_direction < 0 and ppi_direction >= 0:
            phase = "被动补库存 (Passive Restocking)"
            code = "PEAKING"
            desc = "需求开始边际走弱但生产维持高位，产成品库存被动积压，需防范价格承压与利润收缩。"
        elif pmi_direction < 0 and ppi_direction < 0:
            phase = "主动去库存 (Active Destocking)"
            code = "CONTRACTION"
            desc = "企业削减排产并压降原料采购，价格低迷，产业链竞争白热化，现金流稳健龙头胜出。"
        else:
            phase = "被动去库存 (Passive Destocking / Recovery)"
            code = "RECOVERY"
            desc = "下游终端需求率先企稳回升，库存降至历史低位，新一轮补库与设备投资即将启动。"

        return {
            "inventory_phase": phase,
            "phase_code": code,
            "latest_pmi": latest_pmi,
            "latest_ppi": latest_ppi,
            "pmi_delta": round(pmi_direction, 2),
            "ppi_delta": round(ppi_direction, 2),
            "description": desc
        }
