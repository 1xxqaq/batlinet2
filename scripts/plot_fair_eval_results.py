import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


METRICS = ["RMSE", "MAE", "MAPE"]


def parse_args():
    parser = argparse.ArgumentParser(
        description="基于 analyze_fair_eval_results.py 生成的表格绘制公平评估结果图。"
    )
    parser.add_argument(
        "--analysis-dir",
        required=True,
        help="分析输出目录，例如 analysis/fair_eval/mix_20_protocol_v1",
    )
    parser.add_argument(
        "--baseline-name",
        default="batlinet_original",
        help="基线方法名称，需与分析脚本输出列名一致。",
    )
    parser.add_argument(
        "--candidate-name",
        default="batlinet_supervised_weighted",
        help="候选方法名称，需与分析脚本输出列名一致。",
    )
    return parser.parse_args()


def setup_matplotlib():
    plt.rcParams["font.sans-serif"] = [
        "Microsoft YaHei",
        "SimHei",
        "Noto Sans CJK SC",
        "Arial Unicode MS",
        "DejaVu Sans",
    ]
    plt.rcParams["axes.unicode_minus"] = False


def ensure_figure_dir(analysis_dir: Path) -> Path:
    figure_dir = analysis_dir / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    return figure_dir


def resolve_col(df: pd.DataFrame, prefix: str) -> str:
    for col in df.columns:
        if col.startswith(prefix):
            return col
    raise KeyError(f"找不到以 {prefix} 开头的列。")


def plot_metric_mean_bar(
    metric_summary: pd.DataFrame,
    figure_dir: Path,
    baseline_name: str,
    candidate_name: str,
):
    width = 0.35
    main_metrics = ["RMSE", "MAE"]
    small_metric = "MAPE"
    metric_col = resolve_col(metric_summary, "metric")
    baseline_mean_col = resolve_col(metric_summary, "baseline_mean")
    candidate_mean_col = resolve_col(metric_summary, "candidate_mean")

    main_x = np.arange(len(main_metrics))
    main_baseline = [
        metric_summary.loc[metric_summary[metric_col] == m, baseline_mean_col].iloc[0]
        for m in main_metrics
    ]
    main_candidate = [
        metric_summary.loc[metric_summary[metric_col] == m, candidate_mean_col].iloc[0]
        for m in main_metrics
    ]
    small_baseline = metric_summary.loc[metric_summary[metric_col] == small_metric, baseline_mean_col].iloc[0]
    small_candidate = metric_summary.loc[metric_summary[metric_col] == small_metric, candidate_mean_col].iloc[0]

    fig, ax1 = plt.subplots(figsize=(9, 5))
    ax2 = ax1.twinx()

    ax1.bar(
        main_x - width / 2,
        main_baseline,
        width,
        label=baseline_name,
        color="#7A8DA4",
    )
    ax1.bar(
        main_x + width / 2,
        main_candidate,
        width,
        label=candidate_name,
        color="#D98555",
    )
    mape_x = np.array([2.2])
    ax2.bar(
        mape_x - width / 2,
        [small_baseline],
        width,
        color="#7A8DA4",
        alpha=0.9,
    )
    ax2.bar(
        mape_x + width / 2,
        [small_candidate],
        width,
        color="#D98555",
        alpha=0.9,
    )

    all_x = list(main_x) + [float(mape_x[0])]
    ax1.set_xticks(all_x)
    ax1.set_xticklabels(main_metrics + [small_metric])
    ax1.set_ylabel("RMSE / MAE")
    ax2.set_ylabel("MAPE")
    ax1.set_title("指标均值对比")
    ax1.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(figure_dir / "metric_means_bar.png", dpi=200)
    plt.close(fig)


def plot_metric_by_seed(
    seed_metrics: pd.DataFrame,
    figure_dir: Path,
    baseline_name: str,
    candidate_name: str,
):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    seed_col = resolve_col(seed_metrics, "seed")
    seeds = seed_metrics[seed_col].to_numpy()

    for ax, metric in zip(axes, METRICS):
        baseline_col = resolve_col(seed_metrics, f"{baseline_name}_{metric}")
        candidate_col = resolve_col(seed_metrics, f"{candidate_name}_{metric}")
        ax.plot(
            seeds,
            seed_metrics[baseline_col],
            marker="o",
            label=baseline_name,
            color="#7A8DA4",
        )
        ax.plot(
            seeds,
            seed_metrics[candidate_col],
            marker="o",
            label=candidate_name,
            color="#D98555",
        )
        ax.set_title(metric)
        ax.set_xlabel("seed")
        ax.set_ylabel("指标值")
        ax.grid(alpha=0.25)
    axes[0].legend()
    fig.tight_layout()
    fig.savefig(figure_dir / "metrics_by_seed_lines.png", dpi=200)
    plt.close(fig)


def plot_error_delta_hist(sample_level: pd.DataFrame, figure_dir: Path):
    delta_col = resolve_col(sample_level, "delta_abs_error")
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(sample_level[delta_col], bins=40, color="#5B8E7D", edgecolor="white")
    ax.axvline(0.0, color="#B33A3A", linestyle="--", linewidth=1.5)
    ax.set_xlabel("绝对误差差值（改进值 - 原始值）")
    ax.set_ylabel("样本数")
    ax.set_title("样本级误差改善分布")
    fig.tight_layout()
    fig.savefig(figure_dir / "sample_error_delta_hist.png", dpi=200)
    plt.close(fig)


def plot_error_scatter(
    sample_level: pd.DataFrame,
    figure_dir: Path,
    baseline_name: str,
    candidate_name: str,
):
    x = sample_level[resolve_col(sample_level, f"{baseline_name}_abs_error")]
    y = sample_level[resolve_col(sample_level, f"{candidate_name}_abs_error")]

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(x, y, s=18, alpha=0.6, color="#4C78A8")
    lim_min = min(float(x.min()), float(y.min()))
    lim_max = max(float(x.max()), float(y.max()))
    ax.plot([lim_min, lim_max], [lim_min, lim_max], linestyle="--", color="#B33A3A")
    ax.set_xlabel(f"{baseline_name} 绝对误差")
    ax.set_ylabel(f"{candidate_name} 绝对误差")
    ax.set_title("样本级绝对误差对比")
    fig.tight_layout()
    fig.savefig(figure_dir / "sample_error_scatter.png", dpi=200)
    plt.close(fig)


def plot_support_weight_distribution(support_details: pd.DataFrame, figure_dir: Path):
    weight_col = resolve_col(support_details, "support_weight")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    axes[0].hist(
        support_details[weight_col],
        bins=40,
        color="#D98555",
        edgecolor="white",
    )
    axes[0].set_title("参考权重分布")
    axes[0].set_xlabel("权重")
    axes[0].set_ylabel("计数")

    axes[1].boxplot(support_details[weight_col], vert=True)
    axes[1].set_title("参考权重箱线图")
    axes[1].set_ylabel("权重")

    fig.tight_layout()
    fig.savefig(figure_dir / "support_weight_distribution.png", dpi=200)
    plt.close(fig)


def plot_representative_weight_bars(
    representative_support_details: pd.DataFrame,
    figure_dir: Path,
):
    if representative_support_details.empty:
        return

    group_col = resolve_col(representative_support_details, "group")
    seed_col = resolve_col(representative_support_details, "seed")
    test_index_col = resolve_col(representative_support_details, "test_index")
    test_cell_col = resolve_col(representative_support_details, "test_cell_id")
    rank_col = resolve_col(representative_support_details, "support_rank")
    weight_col = resolve_col(representative_support_details, "support_weight")
    pred_col = resolve_col(representative_support_details, "support_prediction")
    target_col = resolve_col(representative_support_details, "target")

    grouped = representative_support_details.groupby(
        [group_col, seed_col, test_index_col, test_cell_col], sort=False
    )
    for (group, seed, test_index, test_cell_id), df in grouped:
        df = df.sort_values(rank_col)
        labels = [f"R{int(v)}" for v in df[rank_col].tolist()]
        weights = df[weight_col].to_numpy()
        support_errors = np.abs(df[pred_col].to_numpy() - df[target_col].to_numpy())

        fig, ax1 = plt.subplots(figsize=(12, 4.8))
        ax2 = ax1.twinx()
        x = np.arange(len(weights))
        ax1.bar(x, weights, color="#D98555", alpha=0.85)
        ax2.plot(x, support_errors, color="#4C78A8", marker="o", linewidth=1.6)
        ax2.invert_yaxis()

        ax1.set_xticks(x)
        ax1.set_xticklabels(labels, rotation=0, fontsize=8)
        ax1.set_ylabel("权重")
        ax2.set_ylabel("单参考绝对误差")
        ax1.set_xlabel("参考电池序号")
        ax1.set_title(f"{group} | seed={seed} | test_idx={test_index} | {test_cell_id}")
        ax1.text(
            0.99,
            0.95,
            "具体 cell_id 见 representative_support_details.csv",
            transform=ax1.transAxes,
            ha="right",
            va="top",
            fontsize=8,
            bbox=dict(boxstyle="round,pad=0.25", facecolor="white", alpha=0.85, edgecolor="#cccccc"),
        )
        bar_proxy = plt.Line2D([0], [0], color="#D98555", linewidth=8)
        line_proxy = plt.Line2D([0], [0], color="#4C78A8", marker="o")
        ax1.legend([bar_proxy, line_proxy], ["参考权重", "单参考绝对误差"], loc="upper left")
        fig.tight_layout()
        fig.savefig(
            figure_dir / f"representative_{group}_seed{seed}_test{test_index}.png",
            dpi=200,
        )
        plt.close(fig)


def plot_branch_mean_by_seed(
    support_details: pd.DataFrame,
    figure_dir: Path,
):
    if support_details.empty:
        return

    seed_col = resolve_col(support_details, "seed")
    test_index_col = resolve_col(support_details, "test_index")
    target_col = resolve_col(support_details, "target")
    y_ori_col = resolve_col(support_details, "y_ori")
    y_sup_agg_col = resolve_col(support_details, "y_sup_agg")

    per_sample = (
        support_details.groupby([seed_col, test_index_col], as_index=False)
        .agg(
            target=(target_col, "first"),
            y_ori=(y_ori_col, "first"),
            y_sup_agg=(y_sup_agg_col, "first"),
        )
        .sort_values([seed_col, test_index_col])
    )

    seed_mean = (
        per_sample.groupby(seed_col, as_index=False)
        .agg(
            target_mean=("target", "mean"),
            y_ori_mean=("y_ori", "mean"),
            y_sup_agg_mean=("y_sup_agg", "mean"),
        )
        .sort_values(seed_col)
    )

    x = np.arange(len(seed_mean))
    width = 0.25

    fig, ax = plt.subplots(figsize=(11, 5))
    ax.bar(x - width, seed_mean["target_mean"], width, label="真实值均值", color="#7A8DA4")
    ax.bar(x, seed_mean["y_ori_mean"], width, label="目标电池分支均值", color="#D98555")
    ax.bar(x + width, seed_mean["y_sup_agg_mean"], width, label="参考电池分支均值", color="#5B8E7D")
    ax.set_xticks(x)
    ax.set_xticklabels(seed_mean[seed_col].tolist())
    ax.set_xlabel("seed")
    ax.set_ylabel("数值")
    ax.set_title("各 seed 的目标分支 / 参考分支 / 真实值均值对比")
    ax.legend()
    fig.tight_layout()
    fig.savefig(figure_dir / "branch_mean_by_seed.png", dpi=200)
    plt.close(fig)


def plot_oracle_aggregation_summary(
    oracle_summary: pd.DataFrame,
    figure_dir: Path,
):
    if oracle_summary.empty:
        return

    method_col = resolve_col(oracle_summary, "method")
    metric_col = resolve_col(oracle_summary, "metric")
    mean_col = resolve_col(oracle_summary, "metric_mean")
    methods = [
        "candidate_weighted",
        "same_support_mean",
        "same_support_median",
        "oracle_top5_mean",
        "oracle_top3_mean",
        "oracle_best",
    ]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8))
    colors = ["#D98555", "#7A8DA4", "#5B8E7D", "#B0A160", "#8A6FB0", "#B33A3A"]
    for ax, metric in zip(axes, METRICS):
        sub = oracle_summary[oracle_summary[metric_col] == metric].copy()
        values = []
        labels = []
        bar_colors = []
        for idx, method in enumerate(methods):
            row = sub[sub[method_col] == method]
            if row.empty:
                continue
            labels.append(method)
            values.append(float(row[mean_col].iloc[0]))
            bar_colors.append(colors[idx])
        ax.bar(np.arange(len(values)), values, color=bar_colors)
        ax.set_xticks(np.arange(len(labels)))
        ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=8)
        ax.set_title(metric)
        ax.set_ylabel("seed 均值")
        ax.grid(axis="y", alpha=0.2)
    fig.suptitle("当前加权与 oracle 聚合上限对比", y=1.02)
    fig.tight_layout()
    fig.savefig(figure_dir / "oracle_aggregation_metric_summary.png", dpi=200)
    plt.close(fig)


def plot_weight_error_alignment(
    alignment: pd.DataFrame,
    summary: pd.DataFrame,
    figure_dir: Path,
):
    if alignment.empty or summary.empty:
        return

    spearman_col = resolve_col(alignment, "spearman_weight_error")
    rank_col = resolve_col(alignment, "highest_weight_error_rank")
    seed_col = resolve_col(summary, "seed")
    top1_col = resolve_col(summary, "top1_error_hit_rate")
    top3_col = resolve_col(summary, "top3_error_hit_rate")
    top5_col = resolve_col(summary, "top5_error_hit_rate")

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))
    axes[0].hist(alignment[spearman_col].dropna(), bins=35, color="#5B8E7D", edgecolor="white")
    axes[0].axvline(0.0, color="#B33A3A", linestyle="--", linewidth=1.3)
    axes[0].set_title("权重-误差 Spearman 相关")
    axes[0].set_xlabel("相关系数")
    axes[0].set_ylabel("样本数")

    axes[1].hist(alignment[rank_col].dropna(), bins=np.arange(1, 35) - 0.5, color="#D98555", edgecolor="white")
    axes[1].set_title("最高权重参考的误差排名")
    axes[1].set_xlabel("排名，1 表示误差最低")
    axes[1].set_ylabel("样本数")

    seed_summary = summary[summary[seed_col].astype(str) != "overall"].copy()
    seeds = seed_summary[seed_col].astype(str).tolist()
    x = np.arange(len(seed_summary))
    width = 0.25
    axes[2].bar(x - width, seed_summary[top1_col], width, label="top1", color="#B33A3A")
    axes[2].bar(x, seed_summary[top3_col], width, label="top3", color="#8A6FB0")
    axes[2].bar(x + width, seed_summary[top5_col], width, label="top5", color="#5B8E7D")
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(seeds)
    axes[2].set_ylim(0, 1)
    axes[2].set_title("最高权重参考命中低误差参考比例")
    axes[2].set_xlabel("seed")
    axes[2].set_ylabel("比例")
    axes[2].legend()
    fig.tight_layout()
    fig.savefig(figure_dir / "weight_error_alignment.png", dpi=200)
    plt.close(fig)


def main():
    args = parse_args()
    setup_matplotlib()

    analysis_dir = Path(args.analysis_dir)
    table_dir = analysis_dir / "tables"
    sample_dir = analysis_dir / "samples"
    figure_dir = ensure_figure_dir(analysis_dir)

    seed_metrics = pd.read_csv(table_dir / "seed_level_metrics.csv")
    metric_summary = pd.read_csv(table_dir / "metric_summary.csv")
    sample_level = pd.read_csv(table_dir / "sample_level_errors.csv")
    support_details = pd.read_csv(sample_dir / "candidate_support_details.csv")
    representative_support_details = pd.read_csv(sample_dir / "representative_support_details.csv")
    oracle_summary_path = table_dir / "oracle_aggregation_metric_summary.csv"
    alignment_path = table_dir / "weight_error_alignment_by_sample.csv"
    alignment_summary_path = table_dir / "weight_error_alignment_summary.csv"

    plot_metric_mean_bar(metric_summary, figure_dir, args.baseline_name, args.candidate_name)
    plot_metric_by_seed(seed_metrics, figure_dir, args.baseline_name, args.candidate_name)
    plot_error_delta_hist(sample_level, figure_dir)
    plot_error_scatter(sample_level, figure_dir, args.baseline_name, args.candidate_name)
    plot_support_weight_distribution(support_details, figure_dir)
    plot_representative_weight_bars(representative_support_details, figure_dir)
    plot_branch_mean_by_seed(support_details, figure_dir)
    if oracle_summary_path.exists():
        plot_oracle_aggregation_summary(pd.read_csv(oracle_summary_path), figure_dir)
    if alignment_path.exists() and alignment_summary_path.exists():
        plot_weight_error_alignment(
            pd.read_csv(alignment_path),
            pd.read_csv(alignment_summary_path),
            figure_dir,
        )

    print(f"绘图完成，输出目录：{figure_dir}")


if __name__ == "__main__":
    main()
