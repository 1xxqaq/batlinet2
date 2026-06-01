import argparse
import io
import pickle
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch


METRICS = ["RMSE", "MAE", "MAPE"]


def parse_args():
    parser = argparse.ArgumentParser(
        description="分析固定测试参考协议下的 BatLiNet 公平评估结果。"
    )
    parser.add_argument(
        "--baseline-workspace",
        required=True,
        help="原始 BatLiNet 公平评估 workspace。",
    )
    parser.add_argument(
        "--candidate-workspace",
        required=True,
        help="候选方法公平评估 workspace，例如 supervised_weighted。",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="分析输出目录。",
    )
    parser.add_argument(
        "--baseline-name",
        default="batlinet_original",
        help="基线方法显示名称。",
    )
    parser.add_argument(
        "--candidate-name",
        default="batlinet_supervised_weighted",
        help="候选方法显示名称。",
    )
    parser.add_argument(
        "--experiment-name",
        default="mix_20_protocol_v1",
        help="实验名称，仅用于输出说明。",
    )
    parser.add_argument(
        "--representative-count-per-group",
        type=int,
        default=3,
        help="每类代表样本保留数量。",
    )
    return parser.parse_args()


def ensure_dirs(output_dir: Path) -> Dict[str, Path]:
    dirs = {
        "root": output_dir,
        "tables": output_dir / "tables",
        "summary": output_dir / "summary",
        "samples": output_dir / "samples",
    }
    for path in dirs.values():
        path.mkdir(parents=True, exist_ok=True)
    return dirs


def latest_prediction_files(workspace: Path) -> Dict[int, Path]:
    by_seed: Dict[int, List[Path]] = {}
    for path in workspace.glob("predictions_seed_*_*.pkl"):
        name = path.stem
        parts = name.split("_")
        try:
            seed_index = parts.index("seed")
            seed = int(parts[seed_index + 1])
        except (ValueError, IndexError):
            continue
        by_seed.setdefault(seed, []).append(path)

    latest = {}
    for seed, paths in by_seed.items():
        latest[seed] = max(paths, key=lambda p: p.stat().st_mtime)
    return latest


def load_pickle(path: Path) -> dict:
    original_load_from_bytes = torch.storage._load_from_bytes

    def cpu_load_from_bytes(b):
        return torch.load(
            io.BytesIO(b),
            map_location=torch.device("cpu"),
            weights_only=False,
        )

    with open(path, "rb") as f:
        try:
            torch.storage._load_from_bytes = cpu_load_from_bytes
            return pickle.load(f)
        finally:
            torch.storage._load_from_bytes = original_load_from_bytes


def tensor_to_numpy(value):
    if value is None:
        return None
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def inverse_prediction_array(data_bundle, array_like):
    if array_like is None:
        return None
    tensor = torch.as_tensor(array_like).detach().cpu().float()
    if data_bundle.label_transformation is not None:
        tensor = data_bundle.label_transformation.inverse_transform(tensor)
    return tensor.numpy()


def get_source_name(source_path: str) -> str:
    parts = Path(source_path).parts
    if "processed" in parts:
        idx = parts.index("processed")
        if idx + 1 < len(parts):
            return parts[idx + 1]
    return "unknown"


def inverse_labels(data_bundle, prediction_tensor: torch.Tensor) -> Tuple[np.ndarray, np.ndarray]:
    target = data_bundle.test_data.label.detach().cpu()
    prediction = prediction_tensor.detach().cpu()
    if data_bundle.label_transformation is not None:
        target = data_bundle.label_transformation.inverse_transform(target)
        prediction = data_bundle.label_transformation.inverse_transform(prediction)
    return target.numpy(), prediction.numpy()


def build_seed_metric_rows(
    baseline_files: Dict[int, Path],
    candidate_files: Dict[int, Path],
    baseline_name: str,
    candidate_name: str,
) -> Tuple[pd.DataFrame, Dict[int, dict], Dict[int, dict]]:
    common_seeds = sorted(set(baseline_files) & set(candidate_files))
    if not common_seeds:
        raise RuntimeError("基线和候选 workspace 没有可对齐的 seed。")

    rows = []
    baseline_objs = {}
    candidate_objs = {}
    for seed in common_seeds:
        baseline_obj = load_pickle(baseline_files[seed])
        candidate_obj = load_pickle(candidate_files[seed])
        baseline_objs[seed] = baseline_obj
        candidate_objs[seed] = candidate_obj

        row = {"seed": seed}
        for metric in METRICS:
            baseline_value = float(baseline_obj["scores"][metric])
            candidate_value = float(candidate_obj["scores"][metric])
            row[f"{baseline_name}_{metric}"] = baseline_value
            row[f"{candidate_name}_{metric}"] = candidate_value
            row[f"delta_{metric}"] = candidate_value - baseline_value
        rows.append(row)

    return pd.DataFrame(rows).sort_values("seed"), baseline_objs, candidate_objs


def build_metric_summary(seed_metrics: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for metric in METRICS:
        baseline_col = [c for c in seed_metrics.columns if c.endswith(f"_{metric}") and not c.startswith("delta_")][0]
        candidate_col = [c for c in seed_metrics.columns if c.endswith(f"_{metric}") and not c.startswith("delta_")][1]
        delta_col = f"delta_{metric}"
        rows.append(
            {
                "metric": metric,
                "baseline_mean": seed_metrics[baseline_col].mean(),
                "baseline_std": seed_metrics[baseline_col].std(ddof=1),
                "candidate_mean": seed_metrics[candidate_col].mean(),
                "candidate_std": seed_metrics[candidate_col].std(ddof=1),
                "delta_mean": seed_metrics[delta_col].mean(),
                "delta_std": seed_metrics[delta_col].std(ddof=1),
            }
        )
    return pd.DataFrame(rows)


def validate_test_alignment(seed: int, baseline_obj: dict, candidate_obj: dict):
    baseline_meta = baseline_obj["metadata"]["test_cells"]
    candidate_meta = candidate_obj["metadata"]["test_cells"]
    if baseline_meta is None or candidate_meta is None:
        raise RuntimeError(f"seed={seed} 缺少 test metadata，无法做样本级对齐。")
    if len(baseline_meta) != len(candidate_meta):
        raise RuntimeError(f"seed={seed} 的测试样本数量不一致。")

    for idx, (b, c) in enumerate(zip(baseline_meta, candidate_meta)):
        if b["cell_id"] != c["cell_id"]:
            raise RuntimeError(
                f"seed={seed} test_idx={idx} 的 cell_id 不一致："
                f"{b['cell_id']} vs {c['cell_id']}"
            )


def build_sample_level_rows(
    baseline_objs: Dict[int, dict],
    candidate_objs: Dict[int, dict],
    baseline_name: str,
    candidate_name: str,
) -> pd.DataFrame:
    rows = []
    for seed in sorted(baseline_objs):
        baseline_obj = baseline_objs[seed]
        candidate_obj = candidate_objs[seed]
        validate_test_alignment(seed, baseline_obj, candidate_obj)

        baseline_target, baseline_pred = inverse_labels(
            baseline_obj["data"], baseline_obj["prediction"]
        )
        candidate_target, candidate_pred = inverse_labels(
            candidate_obj["data"], candidate_obj["prediction"]
        )

        if not np.allclose(baseline_target, candidate_target):
            raise RuntimeError(f"seed={seed} 的真实标签不一致，无法进行样本级对比。")

        test_meta = baseline_obj["metadata"]["test_cells"]
        for test_idx, meta in enumerate(test_meta):
            label = float(baseline_target[test_idx])
            baseline_error = abs(float(baseline_pred[test_idx]) - label)
            candidate_error = abs(float(candidate_pred[test_idx]) - label)
            rows.append(
                {
                    "seed": seed,
                    "test_index": test_idx,
                    "cell_id": meta["cell_id"],
                    "source_path": meta["source_path"],
                    "source_name": get_source_name(meta["source_path"]),
                    "target": label,
                    f"{baseline_name}_prediction": float(baseline_pred[test_idx]),
                    f"{candidate_name}_prediction": float(candidate_pred[test_idx]),
                    f"{baseline_name}_abs_error": baseline_error,
                    f"{candidate_name}_abs_error": candidate_error,
                    "delta_abs_error": candidate_error - baseline_error,
                }
            )
    return pd.DataFrame(rows).sort_values(["seed", "test_index"])


def build_representative_samples(
    sample_level: pd.DataFrame,
    top_k: int,
) -> pd.DataFrame:
    improve = sample_level.nsmallest(top_k, "delta_abs_error").copy()
    improve["group"] = "improve_most"

    stable = sample_level.assign(abs_delta=sample_level["delta_abs_error"].abs())
    stable = stable.nsmallest(top_k, "abs_delta").copy()
    stable["group"] = "change_least"

    regress = sample_level.nlargest(top_k, "delta_abs_error").copy()
    regress["group"] = "regress_most"

    cols = ["group"] + [c for c in sample_level.columns]
    return pd.concat([improve[cols], stable[cols], regress[cols]], ignore_index=True)


def build_candidate_support_detail_rows(
    candidate_objs: Dict[int, dict],
    alpha: float = 0.5,
) -> pd.DataFrame:
    rows = []
    for seed in sorted(candidate_objs):
        obj = candidate_objs[seed]
        diagnostics = obj.get("diagnostics") or {}
        support_index = tensor_to_numpy(diagnostics.get("support_index"))
        support_weight = tensor_to_numpy(diagnostics.get("support_weight"))
        y_sup = inverse_prediction_array(obj["data"], tensor_to_numpy(diagnostics.get("y_sup")))
        y_sup_agg = inverse_prediction_array(obj["data"], tensor_to_numpy(diagnostics.get("y_sup_agg")))
        y_ori = inverse_prediction_array(obj["data"], tensor_to_numpy(diagnostics.get("y_ori")))

        if support_index is None or support_weight is None:
            continue

        target, prediction = inverse_labels(obj["data"], obj["prediction"])
        train_meta = obj["metadata"]["train_cells"]
        test_meta = obj["metadata"]["test_cells"]

        for test_idx, test_cell in enumerate(test_meta):
            for rank_idx, train_idx in enumerate(support_index[test_idx].tolist()):
                train_cell = train_meta[int(train_idx)]
                support_prediction = None if y_sup is None else float(y_sup[test_idx, rank_idx])
                support_fused_prediction = None
                support_fused_abs_error = None
                if y_ori is not None and support_prediction is not None:
                    support_fused_prediction = (
                        (1.0 - alpha) * float(y_ori[test_idx])
                        + alpha * support_prediction
                    )
                    support_fused_abs_error = abs(
                        support_fused_prediction - float(target[test_idx])
                    )
                rows.append(
                    {
                        "seed": seed,
                        "test_index": test_idx,
                        "test_cell_id": test_cell["cell_id"],
                        "test_source_name": get_source_name(test_cell["source_path"]),
                        "target": float(target[test_idx]),
                        "final_prediction": float(prediction[test_idx]),
                        "y_ori": None if y_ori is None else float(y_ori[test_idx]),
                        "y_sup_agg": None if y_sup_agg is None else float(y_sup_agg[test_idx]),
                        "support_rank": rank_idx,
                        "train_index": int(train_idx),
                        "support_cell_id": train_cell["cell_id"],
                        "support_source_path": train_cell["source_path"],
                        "support_source_name": get_source_name(train_cell["source_path"]),
                        "support_weight": float(support_weight[test_idx, rank_idx]),
                        "support_prediction": support_prediction,
                        "support_fused_prediction": support_fused_prediction,
                        "support_fused_abs_error": support_fused_abs_error,
                    }
                )
    return pd.DataFrame(rows)


def annotate_columns(df: pd.DataFrame) -> pd.DataFrame:
    mapping = {
        "seed": "seed（随机种子）",
        "metric": "metric（指标）",
        "test_index": "test_index（测试样本索引）",
        "cell_id": "cell_id（测试电池编号）",
        "source_path": "source_path（测试电池源路径）",
        "source_name": "source_name（测试电池来源）",
        "target": "target（真实值）",
        "group": "group（代表样本分组）",
        "abs_delta": "abs_delta（绝对差值绝对值）",
        "test_cell_id": "test_cell_id（测试电池编号）",
        "test_source_name": "test_source_name（测试电池来源）",
        "final_prediction": "final_prediction（最终融合预测）",
        "y_ori": "y_ori（目标电池分支预测）",
        "y_sup_agg": "y_sup_agg（参考电池分支聚合预测）",
        "support_rank": "support_rank（参考电池序号）",
        "train_index": "train_index（训练集电池索引）",
        "support_cell_id": "support_cell_id（参考电池编号）",
        "support_source_path": "support_source_path（参考电池源路径）",
        "support_source_name": "support_source_name（参考电池来源）",
        "support_weight": "support_weight（参考权重）",
        "support_prediction": "support_prediction（单参考预测）",
        "support_fused_prediction": "support_fused_prediction（单参考融合后预测）",
        "support_fused_abs_error": "support_fused_abs_error（单参考融合后绝对误差）",
        "selected_count": "selected_count（被选次数）",
        "mean_weight": "mean_weight（平均权重）",
        "max_weight": "max_weight（最大权重）",
        "method": "method（聚合方法）",
        "value": "value（指标值）",
        "metric_mean": "metric_mean（seed均值）",
        "metric_std": "metric_std（seed标准差）",
        "candidate_weighted_final": "candidate_weighted_final（当前加权最终预测）",
        "candidate_weighted_abs_error": "candidate_weighted_abs_error（当前加权绝对误差）",
        "same_support_mean_final": "same_support_mean_final（同参考均值最终预测）",
        "same_support_mean_abs_error": "same_support_mean_abs_error（同参考均值绝对误差）",
        "same_support_median_final": "same_support_median_final（同参考中位数最终预测）",
        "same_support_median_abs_error": "same_support_median_abs_error（同参考中位数绝对误差）",
        "oracle_best_final": "oracle_best_final（oracle最佳参考最终预测）",
        "oracle_best_abs_error": "oracle_best_abs_error（oracle最佳参考绝对误差）",
        "oracle_top3_mean_final": "oracle_top3_mean_final（oracle前三均值最终预测）",
        "oracle_top3_mean_abs_error": "oracle_top3_mean_abs_error（oracle前三均值绝对误差）",
        "oracle_top5_mean_final": "oracle_top5_mean_final（oracle前五均值最终预测）",
        "oracle_top5_mean_abs_error": "oracle_top5_mean_abs_error（oracle前五均值绝对误差）",
        "best_support_rank": "best_support_rank（最低误差参考序号）",
        "best_support_prediction": "best_support_prediction（最低误差单参考预测）",
        "best_support_abs_error": "best_support_abs_error（最低参考融合后绝对误差）",
        "spearman_weight_error": "spearman_weight_error（权重与单参考误差Spearman相关）",
        "pearson_weight_error": "pearson_weight_error（权重与单参考误差Pearson相关）",
        "weight_entropy": "weight_entropy（权重归一化熵）",
        "best_weight": "best_weight（最低误差参考权重）",
        "highest_weight_rank": "highest_weight_rank（最高权重参考序号）",
        "highest_weight_error_rank": "highest_weight_error_rank（最高权重参考的误差排名）",
        "highest_weight_abs_error": "highest_weight_abs_error（最高权重参考绝对误差）",
        "spearman_weight_fused_error": "spearman_weight_fused_error（权重与融合后误差Spearman相关）",
        "pearson_weight_fused_error": "pearson_weight_fused_error（权重与融合后误差Pearson相关）",
        "best_fused_weight": "best_fused_weight（最低融合后误差参考权重）",
        "highest_weight_fused_error_rank": "highest_weight_fused_error_rank（最高权重参考的融合后误差排名）",
        "highest_weight_fused_abs_error": "highest_weight_fused_abs_error（最高权重参考融合后绝对误差）",
        "best_support_fused_abs_error": "best_support_fused_abs_error（最低融合后绝对误差）",
        "median_support_fused_abs_error": "median_support_fused_abs_error（融合后误差中位数）",
        "highest_weight_in_top1_fused_error": "highest_weight_in_top1_fused_error（最高权重是否融合后误差第1）",
        "highest_weight_in_top3_fused_error": "highest_weight_in_top3_fused_error（最高权重是否融合后误差前3）",
        "highest_weight_in_top5_fused_error": "highest_weight_in_top5_fused_error（最高权重是否融合后误差前5）",
        "median_support_abs_error": "median_support_abs_error（单参考误差中位数）",
        "highest_weight_in_top1_error": "highest_weight_in_top1_error（最高权重是否误差第1）",
        "highest_weight_in_top3_error": "highest_weight_in_top3_error（最高权重是否误差前3）",
        "highest_weight_in_top5_error": "highest_weight_in_top5_error（最高权重是否误差前5）",
        "sample_count": "sample_count（样本数）",
        "mean_spearman_weight_error": "mean_spearman_weight_error（平均Spearman相关）",
        "median_spearman_weight_error": "median_spearman_weight_error（中位Spearman相关）",
        "mean_spearman_weight_fused_error": "mean_spearman_weight_fused_error（权重与融合后误差平均Spearman相关）",
        "median_spearman_weight_fused_error": "median_spearman_weight_fused_error（权重与融合后误差中位Spearman相关）",
        "positive_spearman_fraction": "positive_spearman_fraction（正相关样本占比）",
        "positive_spearman_fused_fraction": "positive_spearman_fused_fraction（融合后误差正相关样本占比）",
        "top1_error_hit_rate": "top1_error_hit_rate（最高权重命中最低误差比例）",
        "top3_error_hit_rate": "top3_error_hit_rate（最高权重命中误差前3比例）",
        "top5_error_hit_rate": "top5_error_hit_rate（最高权重命中误差前5比例）",
        "top1_fused_error_hit_rate": "top1_fused_error_hit_rate（最高权重命中最低融合后误差比例）",
        "top3_fused_error_hit_rate": "top3_fused_error_hit_rate（最高权重命中融合后误差前3比例）",
        "top5_fused_error_hit_rate": "top5_fused_error_hit_rate（最高权重命中融合后误差前5比例）",
        "mean_weight_entropy": "mean_weight_entropy（平均权重归一化熵）",
        "mean_max_weight": "mean_max_weight（平均最大权重）",
        "top1_count": "top1_count（rank0次数）",
        "baseline_mean": "baseline_mean（基线均值）",
        "baseline_std": "baseline_std（基线标准差）",
        "candidate_mean": "candidate_mean（候选均值）",
        "candidate_std": "candidate_std（候选标准差）",
        "delta_mean": "delta_mean（候选减基线均值）",
        "delta_std": "delta_std（候选减基线标准差）",
        "improvement_percent": "improvement_percent（相对基线改善百分比）",
    }

    renamed = {}
    for col in df.columns:
        if col in mapping:
            renamed[col] = mapping[col]
            continue
        if col.startswith("delta_"):
            renamed[col] = f"{col}（候选减基线差值）"
        elif col.endswith("_prediction"):
            renamed[col] = f"{col}（预测值）"
        elif col.endswith("_abs_error"):
            renamed[col] = f"{col}（绝对误差）"
        elif any(col.endswith(f"_{m}") for m in METRICS):
            renamed[col] = f"{col}（指标值）"
        else:
            renamed[col] = col
    return df.rename(columns=renamed)


def build_overall_metric_comparison(metric_summary: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for metric in METRICS:
        row = metric_summary.loc[metric_summary["metric"] == metric].iloc[0]
        baseline_mean = float(row["baseline_mean"])
        candidate_mean = float(row["candidate_mean"])
        delta_mean = float(row["delta_mean"])
        improvement_percent = (
            (baseline_mean - candidate_mean) / baseline_mean * 100.0
            if baseline_mean != 0
            else np.nan
        )
        rows.append(
            {
                "metric": metric,
                "baseline_mean": baseline_mean,
                "candidate_mean": candidate_mean,
                "delta_mean": delta_mean,
                "improvement_percent": improvement_percent,
            }
        )
    return pd.DataFrame(rows)


def build_support_usage_tables(support_details: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    by_train_cell = (
        support_details.groupby(
            ["support_cell_id", "support_source_name", "train_index"], as_index=False
        )
        .agg(
            selected_count=("support_cell_id", "size"),
            mean_weight=("support_weight", "mean"),
            max_weight=("support_weight", "max"),
            top1_count=("support_rank", lambda x: int((x == 0).sum())),
        )
        .sort_values(["mean_weight", "selected_count"], ascending=[False, False])
    )

    by_source = (
        support_details.groupby("support_source_name", as_index=False)
        .agg(
            selected_count=("support_source_name", "size"),
            mean_weight=("support_weight", "mean"),
            max_weight=("support_weight", "max"),
        )
        .sort_values(["mean_weight", "selected_count"], ascending=[False, False])
    )
    return by_train_cell, by_source


def metric_values(target: np.ndarray, prediction: np.ndarray) -> Dict[str, float]:
    error = prediction - target
    return {
        "RMSE": float(np.sqrt(np.mean(error ** 2))),
        "MAE": float(np.mean(np.abs(error))),
        "MAPE": float(np.mean(np.abs(error / target))),
    }


def rank_average(values: np.ndarray) -> np.ndarray:
    return pd.Series(values).rank(method="average").to_numpy()


def safe_corr(left: np.ndarray, right: np.ndarray) -> float:
    if len(left) < 2 or np.std(left) == 0 or np.std(right) == 0:
        return np.nan
    return float(np.corrcoef(left, right)[0, 1])


def build_oracle_aggregation_tables(
    support_details: pd.DataFrame,
    alpha: float = 0.5,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    sample_rows = []
    metric_rows = []
    methods = [
        ("candidate_weighted", "candidate_weighted_final"),
        ("same_support_mean", "same_support_mean_final"),
        ("same_support_median", "same_support_median_final"),
        ("oracle_best", "oracle_best_final"),
        ("oracle_top3_mean", "oracle_top3_mean_final"),
        ("oracle_top5_mean", "oracle_top5_mean_final"),
    ]

    for (seed, test_index), group in support_details.groupby(["seed", "test_index"]):
        group = group.sort_values("support_rank")
        target = float(group["target"].iloc[0])
        y_ori = float(group["y_ori"].iloc[0])
        final_prediction = float(group["final_prediction"].iloc[0])
        support_predictions = group["support_prediction"].to_numpy(dtype=float)

        def fuse(support_value):
            return (1.0 - alpha) * y_ori + alpha * float(support_value)

        support_fused_predictions = np.array(
            [fuse(value) for value in support_predictions],
            dtype=float,
        )
        support_fused_errors = np.abs(support_fused_predictions - target)
        order = np.argsort(support_fused_errors)

        row = {
            "seed": int(seed),
            "test_index": int(test_index),
            "test_cell_id": group["test_cell_id"].iloc[0],
            "test_source_name": group["test_source_name"].iloc[0],
            "target": target,
            "y_ori": y_ori,
            "candidate_weighted_final": final_prediction,
            "candidate_weighted_abs_error": abs(final_prediction - target),
            "same_support_mean_final": fuse(np.mean(support_predictions)),
            "same_support_median_final": fuse(np.median(support_predictions)),
            "oracle_best_final": fuse(support_predictions[order[0]]),
            "oracle_top3_mean_final": fuse(np.mean(support_predictions[order[:3]])),
            "oracle_top5_mean_final": fuse(np.mean(support_predictions[order[:5]])),
            "best_support_rank": int(order[0]),
            "best_support_prediction": float(support_predictions[order[0]]),
            "best_support_abs_error": float(support_fused_errors[order[0]]),
        }
        for _, prediction_col in methods[1:]:
            row[prediction_col.replace("_final", "_abs_error")] = abs(
                row[prediction_col] - target
            )
        sample_rows.append(row)

    oracle_sample_level = pd.DataFrame(sample_rows).sort_values(["seed", "test_index"])

    for seed, seed_df in oracle_sample_level.groupby("seed"):
        target = seed_df["target"].to_numpy(dtype=float)
        for method, prediction_col in methods:
            values = metric_values(target, seed_df[prediction_col].to_numpy(dtype=float))
            for metric, value in values.items():
                metric_rows.append(
                    {
                        "seed": int(seed),
                        "method": method,
                        "metric": metric,
                        "value": value,
                    }
                )

    oracle_by_seed = pd.DataFrame(metric_rows).sort_values(["method", "metric", "seed"])
    oracle_summary = (
        oracle_by_seed.groupby(["method", "metric"], as_index=False)
        .agg(
            metric_mean=("value", "mean"),
            metric_std=("value", lambda x: x.std(ddof=1)),
        )
        .sort_values(["metric", "metric_mean"])
    )
    return oracle_sample_level, oracle_by_seed, oracle_summary


def build_weight_error_alignment_tables(
    support_details: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    for (seed, test_index), group in support_details.groupby(["seed", "test_index"]):
        group = group.sort_values("support_rank")
        weights = group["support_weight"].to_numpy(dtype=float)
        predictions = group["support_prediction"].to_numpy(dtype=float)
        target = float(group["target"].iloc[0])
        errors = np.abs(predictions - target)
        if (
            "support_fused_abs_error" in group.columns
            and group["support_fused_abs_error"].notna().all()
        ):
            fused_errors = group["support_fused_abs_error"].to_numpy(dtype=float)
        else:
            fused_errors = errors
        best_error = float(errors.min())
        best_fused_error = float(fused_errors.min())
        highest_weight_index = int(np.argmax(weights))
        error_order = np.argsort(errors)
        error_rank = int(np.where(error_order == highest_weight_index)[0][0]) + 1
        fused_error_order = np.argsort(fused_errors)
        fused_error_rank = int(
            np.where(fused_error_order == highest_weight_index)[0][0]
        ) + 1
        entropy = -float(np.sum(weights * np.log(weights + 1e-12)) / np.log(len(weights)))

        rows.append(
            {
                "seed": int(seed),
                "test_index": int(test_index),
                "test_cell_id": group["test_cell_id"].iloc[0],
                "test_source_name": group["test_source_name"].iloc[0],
                "target": target,
                "spearman_weight_error": safe_corr(
                    rank_average(weights),
                    rank_average(errors),
                ),
                "pearson_weight_error": safe_corr(weights, errors),
                "spearman_weight_fused_error": safe_corr(
                    rank_average(weights),
                    rank_average(fused_errors),
                ),
                "pearson_weight_fused_error": safe_corr(weights, fused_errors),
                "weight_entropy": entropy,
                "max_weight": float(weights.max()),
                "best_weight": float(weights[error_order[0]]),
                "best_fused_weight": float(weights[fused_error_order[0]]),
                "highest_weight_rank": highest_weight_index,
                "highest_weight_error_rank": error_rank,
                "highest_weight_fused_error_rank": fused_error_rank,
                "highest_weight_abs_error": float(errors[highest_weight_index]),
                "highest_weight_fused_abs_error": float(fused_errors[highest_weight_index]),
                "best_support_abs_error": best_error,
                "best_support_fused_abs_error": best_fused_error,
                "median_support_abs_error": float(np.median(errors)),
                "median_support_fused_abs_error": float(np.median(fused_errors)),
                "highest_weight_in_top1_error": int(error_rank <= 1),
                "highest_weight_in_top3_error": int(error_rank <= 3),
                "highest_weight_in_top5_error": int(error_rank <= 5),
                "highest_weight_in_top1_fused_error": int(fused_error_rank <= 1),
                "highest_weight_in_top3_fused_error": int(fused_error_rank <= 3),
                "highest_weight_in_top5_fused_error": int(fused_error_rank <= 5),
            }
        )

    alignment = pd.DataFrame(rows).sort_values(["seed", "test_index"])
    summary = (
        alignment.groupby("seed", as_index=False)
        .agg(
            sample_count=("test_index", "size"),
            mean_spearman_weight_error=("spearman_weight_error", "mean"),
            median_spearman_weight_error=("spearman_weight_error", "median"),
            mean_spearman_weight_fused_error=("spearman_weight_fused_error", "mean"),
            median_spearman_weight_fused_error=("spearman_weight_fused_error", "median"),
            positive_spearman_fraction=("spearman_weight_error", lambda x: float((x > 0).mean())),
            positive_spearman_fused_fraction=(
                "spearman_weight_fused_error", lambda x: float((x > 0).mean())
            ),
            top1_error_hit_rate=("highest_weight_in_top1_error", "mean"),
            top3_error_hit_rate=("highest_weight_in_top3_error", "mean"),
            top5_error_hit_rate=("highest_weight_in_top5_error", "mean"),
            top1_fused_error_hit_rate=("highest_weight_in_top1_fused_error", "mean"),
            top3_fused_error_hit_rate=("highest_weight_in_top3_fused_error", "mean"),
            top5_fused_error_hit_rate=("highest_weight_in_top5_fused_error", "mean"),
            mean_weight_entropy=("weight_entropy", "mean"),
            mean_max_weight=("max_weight", "mean"),
        )
        .sort_values("seed")
    )
    overall = {
        "seed": "overall",
        "sample_count": int(alignment["test_index"].count()),
        "mean_spearman_weight_error": alignment["spearman_weight_error"].mean(),
        "median_spearman_weight_error": alignment["spearman_weight_error"].median(),
        "mean_spearman_weight_fused_error": alignment["spearman_weight_fused_error"].mean(),
        "median_spearman_weight_fused_error": alignment["spearman_weight_fused_error"].median(),
        "positive_spearman_fraction": float((alignment["spearman_weight_error"] > 0).mean()),
        "positive_spearman_fused_fraction": float(
            (alignment["spearman_weight_fused_error"] > 0).mean()
        ),
        "top1_error_hit_rate": alignment["highest_weight_in_top1_error"].mean(),
        "top3_error_hit_rate": alignment["highest_weight_in_top3_error"].mean(),
        "top5_error_hit_rate": alignment["highest_weight_in_top5_error"].mean(),
        "top1_fused_error_hit_rate": alignment["highest_weight_in_top1_fused_error"].mean(),
        "top3_fused_error_hit_rate": alignment["highest_weight_in_top3_fused_error"].mean(),
        "top5_fused_error_hit_rate": alignment["highest_weight_in_top5_fused_error"].mean(),
        "mean_weight_entropy": alignment["weight_entropy"].mean(),
        "mean_max_weight": alignment["max_weight"].mean(),
    }
    summary = pd.concat([summary, pd.DataFrame([overall])], ignore_index=True)
    return alignment, summary


def build_representative_support_details(
    representative_samples: pd.DataFrame,
    support_details: pd.DataFrame,
) -> pd.DataFrame:
    keys = representative_samples[["seed", "test_index", "group"]].copy()
    merged = support_details.merge(keys, on=["seed", "test_index"], how="inner")
    return merged.sort_values(["group", "seed", "test_index", "support_rank"])


def write_summary(
    output_path: Path,
    experiment_name: str,
    baseline_name: str,
    candidate_name: str,
    seed_metrics: pd.DataFrame,
    representative_samples: pd.DataFrame,
):
    lines = []
    lines.append(f"实验名称：{experiment_name}")
    lines.append(f"基线方法：{baseline_name}")
    lines.append(f"候选方法：{candidate_name}")
    lines.append(f"seed 数量：{len(seed_metrics)}")
    lines.append("")
    lines.append("输出文件说明：")
    lines.append("- tables/seed_level_metrics.csv")
    lines.append("  作用：记录每个 seed 下原始 BatLiNet 与 supervised_weighted 的 RMSE、MAE、MAPE，以及两者差值。")
    lines.append("- tables/metric_summary.csv")
    lines.append("  作用：汇总各指标在 8 个 seed 上的均值、标准差和平均差值。")
    lines.append("- tables/sample_level_errors.csv")
    lines.append("  作用：记录每个测试样本在两种方法下的预测值、绝对误差和误差差值，用于样本级分析。")
    lines.append("- tables/representative_samples.csv")
    lines.append("  作用：筛出代表样本，分为提升最大、变化最小、退步最大三类。")
    lines.append("- tables/candidate_support_usage_by_train_cell.csv")
    lines.append("  作用：统计 supervised_weighted 中各训练电池被作为参考电池使用的次数、平均权重、最大权重等。")
    lines.append("- tables/candidate_support_usage_by_source.csv")
    lines.append("  作用：按数据来源统计参考电池的使用情况和权重分布。")
    lines.append("- tables/oracle_aggregation_metrics_by_seed.csv")
    lines.append("  作用：使用同一批测试参考预测，比较当前加权、均值、中位数和 oracle 聚合上限的 seed 级指标。")
    lines.append("- tables/oracle_aggregation_metric_summary.csv")
    lines.append("  作用：汇总 oracle 聚合上限在各指标上的 seed 均值和标准差。")
    lines.append("- tables/weight_error_alignment_by_sample.csv")
    lines.append("  作用：逐样本统计参考权重与单参考误差的相关性，以及最高权重参考是否命中低误差参考。")
    lines.append("- tables/weight_error_alignment_summary.csv")
    lines.append("  作用：按 seed 汇总权重-误差相关性、命中率、权重熵和最大权重。")
    lines.append("- samples/candidate_support_details.csv")
    lines.append("  作用：保存 supervised_weighted 对所有测试样本的参考电池明细，包括参考索引、参考 cell_id、权重、单参考预测等。")
    lines.append("- samples/oracle_aggregation_sample_level.csv")
    lines.append("  作用：逐样本保存当前加权、同参考均值/中位数、oracle 最佳参考、oracle top-k 参考的最终预测和误差。")
    lines.append("- samples/representative_support_details.csv")
    lines.append("  作用：仅保留代表样本的参考电池明细，便于后续单样本绘图和案例分析。")
    lines.append("- summary/analysis_summary.txt")
    lines.append("  作用：说明当前分析目录中各文件的用途，方便后续新对话接手。")
    lines.append("")
    lines.append("图片文件说明：")
    lines.append("- figures/metric_means_bar.png")
    lines.append("  作用：对比两种方法在 RMSE、MAE、MAPE 上的平均指标；其中 MAPE 会单独使用自己的纵轴或子图。")
    lines.append("- figures/metrics_by_seed_lines.png")
    lines.append("  作用：展示每个 seed 下三项指标的变化，用于观察提升是否稳定。")
    lines.append("- figures/sample_error_delta_hist.png")
    lines.append("  作用：展示样本级绝对误差差值分布，判断整体上是更多样本变好还是变差。")
    lines.append("- figures/sample_error_scatter.png")
    lines.append("  作用：对比每个测试样本在两种方法下的绝对误差，低于对角线表示改进。")
    lines.append("- figures/support_weight_distribution.png")
    lines.append("  作用：展示 supervised_weighted 学到的参考权重总体分布。")
    lines.append("- figures/oracle_aggregation_metric_summary.png")
    lines.append("  作用：展示当前加权与 oracle 聚合上限之间的指标差距。")
    lines.append("- figures/weight_error_alignment.png")
    lines.append("  作用：展示权重-误差相关性分布和最高权重参考的低误差命中率。")
    lines.append("- figures/representative_*.png")
    lines.append("  作用：展示代表样本的 32 个参考权重条形图；横轴使用统一参考序号，具体 cell_id 对应关系见 representative_support_details.csv。")
    lines.append("")
    lines.append("代表样本分组说明：")
    for group in ["improve_most", "change_least", "regress_most"]:
        count = int((representative_samples["group"] == group).sum())
        if group == "improve_most":
            group_name = "提升最大"
        elif group == "change_least":
            group_name = "变化最小"
        else:
            group_name = "退步最大"
        lines.append(f"- {group_name}：{count} 个样本")
    output_path.write_text("\n".join(lines), encoding="utf-8")


def main():
    args = parse_args()
    baseline_workspace = Path(args.baseline_workspace)
    candidate_workspace = Path(args.candidate_workspace)
    output_dir = Path(args.output_dir)
    dirs = ensure_dirs(output_dir)

    baseline_files = latest_prediction_files(baseline_workspace)
    candidate_files = latest_prediction_files(candidate_workspace)

    seed_metrics, baseline_objs, candidate_objs = build_seed_metric_rows(
        baseline_files,
        candidate_files,
        args.baseline_name,
        args.candidate_name,
    )
    metric_summary = build_metric_summary(seed_metrics)
    sample_level = build_sample_level_rows(
        baseline_objs,
        candidate_objs,
        args.baseline_name,
        args.candidate_name,
    )
    representative_samples = build_representative_samples(
        sample_level,
        top_k=args.representative_count_per_group,
    )
    support_details = build_candidate_support_detail_rows(candidate_objs)
    usage_by_train_cell, usage_by_source = build_support_usage_tables(support_details)
    representative_support_details = build_representative_support_details(
        representative_samples,
        support_details,
    )
    overall_metric_comparison = build_overall_metric_comparison(metric_summary)
    oracle_sample_level, oracle_by_seed, oracle_summary = build_oracle_aggregation_tables(
        support_details
    )
    weight_error_alignment, weight_error_summary = build_weight_error_alignment_tables(
        support_details
    )

    annotate_columns(seed_metrics).to_csv(
        dirs["tables"] / "seed_level_metrics.csv", index=False, encoding="utf-8-sig"
    )
    annotate_columns(metric_summary).to_csv(
        dirs["tables"] / "metric_summary.csv", index=False, encoding="utf-8-sig"
    )
    annotate_columns(overall_metric_comparison).to_csv(
        dirs["tables"] / "overall_metric_comparison.csv", index=False, encoding="utf-8-sig"
    )
    annotate_columns(sample_level).to_csv(
        dirs["tables"] / "sample_level_errors.csv", index=False, encoding="utf-8-sig"
    )
    annotate_columns(representative_samples).to_csv(
        dirs["tables"] / "representative_samples.csv", index=False, encoding="utf-8-sig"
    )
    annotate_columns(support_details).to_csv(
        dirs["samples"] / "candidate_support_details.csv", index=False, encoding="utf-8-sig"
    )
    annotate_columns(oracle_sample_level).to_csv(
        dirs["samples"] / "oracle_aggregation_sample_level.csv",
        index=False,
        encoding="utf-8-sig",
    )
    annotate_columns(oracle_by_seed).to_csv(
        dirs["tables"] / "oracle_aggregation_metrics_by_seed.csv",
        index=False,
        encoding="utf-8-sig",
    )
    annotate_columns(oracle_summary).to_csv(
        dirs["tables"] / "oracle_aggregation_metric_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )
    annotate_columns(weight_error_alignment).to_csv(
        dirs["tables"] / "weight_error_alignment_by_sample.csv",
        index=False,
        encoding="utf-8-sig",
    )
    annotate_columns(weight_error_summary).to_csv(
        dirs["tables"] / "weight_error_alignment_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )
    annotate_columns(usage_by_train_cell).to_csv(
        dirs["tables"] / "candidate_support_usage_by_train_cell.csv",
        index=False,
        encoding="utf-8-sig",
    )
    annotate_columns(usage_by_source).to_csv(
        dirs["tables"] / "candidate_support_usage_by_source.csv",
        index=False,
        encoding="utf-8-sig",
    )
    annotate_columns(representative_support_details).to_csv(
        dirs["samples"] / "representative_support_details.csv",
        index=False,
        encoding="utf-8-sig",
    )
    write_summary(
        dirs["summary"] / "analysis_summary.txt",
        args.experiment_name,
        args.baseline_name,
        args.candidate_name,
        seed_metrics,
        representative_samples,
    )

    print(f"分析完成，输出目录：{output_dir}")


if __name__ == "__main__":
    main()
