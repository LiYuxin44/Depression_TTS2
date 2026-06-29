#!/usr/bin/env python
"""
分析严重度趋势一致性、汇总 ordinal 回归效果，并输出显著特征列表
usage: python analyze_trends.py --input-severity severity_aggregated.csv --input-ordinal ordinal_regression_summary.csv --out-dir analysis_exp_A/results
"""
from __future__ import annotations

import argparse
import pathlib
from typing import Dict, List

import numpy as np
import pandas as pd
from scipy.stats import linregress, spearmanr
import matplotlib.pyplot as plt
import seaborn as sns


def compute_trend_metrics(group: pd.DataFrame, feature: str) -> Dict[str, float]:
    """对单个 speaker/feature 计算趋势指标。"""
    x = group["severity_level"].to_numpy(dtype=float)
    y = group[feature].to_numpy(dtype=float)

    lr = linregress(x, y)
    rho, p_spearman = spearmanr(x, y)

    diffs = np.diff(y)
    sign_flips = np.sum(np.diff(np.sign(diffs[np.nonzero(diffs)])) != 0) if diffs.size else 0

    return {
        "slope": lr.slope,
        "slope_p": lr.pvalue,
        "spearman_rho": rho,
        "spearman_p": p_spearman,
        "sign_flips": float(sign_flips),
        "range_delta": float(y[-1] - y[0]) if len(y) > 1 else 0.0,
    }


def summarize_trends(severity_df: pd.DataFrame, features: List[str]) -> pd.DataFrame:
    """遍历所有 speaker/feature，计算趋势一致性指标。"""
    records = []
    for speaker_id, grp in severity_df.groupby("speaker_id"):
        grp_sorted = grp.sort_values("severity_level")
        for feature in features:
            if feature not in grp_sorted.columns:
                continue
            metrics = compute_trend_metrics(grp_sorted, feature)
            metrics.update({
                "speaker_id": speaker_id,
                "feature": feature,
                "n_levels": grp_sorted.shape[0],
            })
            records.append(metrics)
    trends_df = pd.DataFrame(records)

    # 添加一致性标签
    trends_df["consistent"] = (
        (trends_df["slope_p"] < 0.05)
        & (trends_df["sign_flips"] == 0)
        & (np.abs(trends_df["spearman_rho"]) >= 0.6)
    )
    return trends_df


def summarize_ordinal(df: pd.DataFrame) -> pd.DataFrame:
    """
    ordinal_regression_summary.csv 已包含 speaker/feature/coef/pseudo_r2。
    这里做一次广义 FDR 校正与阈值标注。
    """
    out = df.copy()
    # 只有 pseudo_r2，可用阈值或自定义评分。若有 p 值可在此处接入。
    out["strong_fit"] = out["ordinal_pseudo_r2"] >= 0.02
    return out


def merge_results(trends: pd.DataFrame, ordinal: pd.DataFrame, out_dir: pathlib.Path) -> pd.DataFrame:
    merged = trends.merge(
        ordinal,
        on=["speaker_id", "feature"],
        how="left",
        suffixes=("_trend", "_ordinal"),
    )

    # 判定趋势方向是否一致
    merged["sign_agree"] = np.sign(merged["slope"]) == np.sign(merged["ordinal_coef"])

    # 添加多层显著性标签
    merged["significance_level"] = "not_significant"

    # Level 1: trend only
    trend_strong = (
        merged["consistent"]
        & (np.abs(merged["range_delta"]) > 0)
    )
    merged.loc[trend_strong, "significance_level"] = "trend_only"

    # Level 2: trend + ordinal拟合度
    with_ordinal = trend_strong & (merged["ordinal_pseudo_r2"].fillna(0) > 0.01)
    merged.loc[with_ordinal, "significance_level"] = "trend+ordinal"

    # Level 3: trend + ordinal + 拟合方向一致
    full_criteria = with_ordinal & merged["sign_agree"].fillna(True)
    merged.loc[full_criteria, "significance_level"] = "strong"

    # 最终布尔标签（原来用的）
    merged["significant_change"] = merged["significance_level"] == "strong"

    # 输出各层级结果
    for level in ["trend_only", "trend+ordinal", "strong"]:
        subset = merged[merged["significance_level"] == level]
        out_file = out_dir / f"significant_features_{level}.csv"
        subset.to_csv(out_file, index=False)
        print(f"输出显著性等级 [{level}]: {len(subset)} 条 → {out_file}")

    return merged



def sanitize_filename(name: str) -> str:
    return "".join(c if c.isalnum() or c in ("-", "_") else "_" for c in name)


def select_features_to_plot(merged: pd.DataFrame, top_n: int) -> List[str]:
    if merged.empty or top_n <= 0:
        return []

    # 先取显著特征，根据 pseudo_r2 与 |slope| 排序
    significant = merged[merged["significant_change"]].copy()
    if not significant.empty:
        significant = significant.assign(priority=-np.abs(significant["slope"]).rank(method="dense", ascending=False))
        significant = significant.sort_values(["ordinal_pseudo_r2", "priority"], ascending=[False, True])
        feature_list = significant["feature"].drop_duplicates().tolist()
    else:
        feature_list = []

    if len(feature_list) < top_n:
        # 补充整体排序靠前的特征
        fallback = (
            merged.assign(priority=-np.abs(merged["slope"]).rank(method="dense", ascending=False))
            .sort_values(["ordinal_pseudo_r2", "priority"], ascending=[False, True])
        )
        feature_list += [f for f in fallback["feature"].tolist() if f not in feature_list]

    return feature_list[:top_n]


def plot_feature_trend(
    severity_df: pd.DataFrame,
    merged: pd.DataFrame,
    feature: str,
    out_dir: pathlib.Path,
):
    if feature not in severity_df.columns:
        return

    out_dir.mkdir(parents=True, exist_ok=True)

    feature_df = severity_df[["speaker_id", "severity_level", "severity_label", feature]].copy()
    feature_df = feature_df.dropna(subset=[feature])
    if feature_df.empty:
        return

    per_speaker = (
        feature_df.groupby(["speaker_id", "severity_level"])[feature]
        .mean()
        .reset_index()
        .sort_values(["speaker_id", "severity_level"])
    )
    per_speaker["baseline"] = per_speaker.groupby("speaker_id")[feature].transform("first")
    per_speaker["delta"] = per_speaker[feature] - per_speaker["baseline"]

    # 若有 speaker 只有一个严重度点，delta 全为 0 也保留
    per_speaker_delta = per_speaker.dropna(subset=["delta"])
    if per_speaker_delta.empty:
        per_speaker_delta = per_speaker.assign(delta=0.0)

    label_map = (
        feature_df[["severity_level", "severity_label"]]
        .drop_duplicates()
        .sort_values("severity_level")
        .set_index("severity_level")
    )
    summary = (
        per_speaker_delta.groupby("severity_level")["delta"]
        .agg(["mean", "std", "count"])
        .reset_index()
        .sort_values("severity_level")
    )
    summary["label"] = summary["severity_level"].map(label_map["severity_label"])

    fig, ax = plt.subplots(figsize=(5, 3.2))

    # 绘制均值+标准差带
    ax.plot(
        summary["severity_level"],
        summary["mean"],
        color="#1f77b4",
        linewidth=2.0,
        label="Mean",
    )

    labels = summary["label"].tolist()
    ax.set_xticks(summary["severity_level"])
    ax.set_xticklabels(labels, rotation=15)
    ax.set_xlabel("Severity")
    ax.set_ylabel(f"Δ{feature} (vs baseline)")
    ax.set_title(f"{feature} delta across severity")

    # 注释 slope / pseudo_r2
    merged_feature = merged[merged["feature"] == feature]
    if not merged_feature.empty:
        mean_slope = merged_feature["slope"].mean()
        pseudo_median = merged_feature["ordinal_pseudo_r2"].median()
        sig_ratio = merged_feature["significant_change"].mean()
        ax.text(
            0.02,
            0.98,
            f"avg slope={mean_slope:.4f}\nmedian pseudo R²={pseudo_median:.3f}\n% significant={sig_ratio*100:.1f}%",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=9,
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.7),
        )

    ax.grid(True, linestyle="--", alpha=0.3)

    filename = sanitize_filename(feature)
    fig.tight_layout()
    fig.savefig(out_dir / f"{filename}.png", dpi=200)
    plt.close(fig)


def plot_selected_features(
    severity_df: pd.DataFrame,
    merged: pd.DataFrame,
    features: List[str],
    out_dir: pathlib.Path,
):
    for feature in features:
        plot_feature_trend(severity_df, merged, feature, out_dir)


def plot_feature_heatmap(
    severity_df: pd.DataFrame,
    features: List[str],
    out_dir: pathlib.Path,
    zscore_axis: int = 1,
):
    if not features:
        return

    out_dir.mkdir(parents=True, exist_ok=True)

    cols = ["speaker_id", "severity_level", "severity_label", *features]
    data = severity_df[cols].copy()
    data = data.dropna(subset=features, how="all")
    if data.empty:
        return

    data = data.sort_values(["speaker_id", "severity_level"])
    melt = data.melt(
        id_vars=["speaker_id", "severity_level", "severity_label"],
        value_vars=features,
        var_name="feature",
        value_name="value",
    )
    melt = melt.dropna(subset=["value"])
    if melt.empty:
        return

    melt["baseline"] = melt.groupby(["speaker_id", "feature"])["value"].transform("first")
    melt["delta"] = melt["value"] - melt["baseline"]

    pivot = (
        melt.groupby(["severity_level", "severity_label", "feature"])["delta"]
        .mean()
        .reset_index()
    )
    table = pivot.pivot(index="severity_level", columns="feature", values="delta")
    table = table.loc[sorted(table.index)]

    if zscore_axis in (0, 1):
        table = table.apply(
            lambda x: (x - x.mean()) / (x.std() + 1e-6),
            axis=zscore_axis,
        )

    severity_labels = (
        pivot.loc[:, ["severity_level", "severity_label"]]
        .drop_duplicates()
        .set_index("severity_level")
        .loc[table.index, "severity_label"]
        .tolist()
    )

    width = max(6, 0.4 * len(table.columns))
    height = max(4, 0.7 * len(table.index))
    plt.figure(figsize=(width, height))
    sns.heatmap(
        table,
        cmap="coolwarm",
        center=0,
        cbar_kws={"label": "Δ (z-score)"},
        linewidths=0.5,
        linecolor="white",
    )
    plt.yticks(np.arange(len(severity_labels)) + 0.5, severity_labels, rotation=0)
    plt.xticks(rotation=45, ha="right")
    plt.xlabel("Feature")
    plt.ylabel("Severity")
    plt.title("Mean Δ per severity (z-score)")
    plt.tight_layout()
    plt.savefig(out_dir / "feature_heatmap.png", dpi=200)
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-severity", required=True)
    parser.add_argument("--input-ordinal", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument(
        "--feature-list",
        help="若只关注部分特征，可提供一个 txt，每行一个特征名称。",
    )
    parser.add_argument(
        "--plot-top-n",
        type=int,
        default=5,
        help="绘制趋势图的特征数量（默认 5）",
    )
    parser.add_argument(
        "--plot-all",
        action="store_true",
        help="若指定，则为每个特征输出趋势图",
    )
    args = parser.parse_args()

    out_dir = pathlib.Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    severity_df = pd.read_csv(args.input_severity)
    ordinal_df = pd.read_csv(args.input_ordinal)

    all_features = [c for c in severity_df.columns if c not in ("speaker_id", "severity_level", "severity_label")]
    if args.feature_list:
        with open(args.feature_list) as f:
            features = [line.strip() for line in f if line.strip()]
        missing = sorted(set(features) - set(all_features))
        if missing:
            raise ValueError(f"Feature(s) not found in severity file: {missing}")
    else:
        features = all_features
    heatmap_features = all_features

    trends = summarize_trends(severity_df[["speaker_id", "severity_level", *features]], features)
    trends.to_csv(out_dir / "trend_metrics.csv", index=False)

    ordinal_summary = summarize_ordinal(ordinal_df)
    ordinal_summary.to_csv(out_dir / "ordinal_summary_aug.csv", index=False)

    merged = merge_results(trends, ordinal_summary, out_dir)
    merged.to_csv(out_dir / "trend_plus_ordinal.csv", index=False)

    # 输出显著特征列表
    significant = merged[merged["significant_change"]]
    significant.to_csv(out_dir / "significant_features.csv", index=False)

    # 绘图
    if args.plot_all:
        plot_features = features
    else:
        plot_features = select_features_to_plot(merged, args.plot_top_n)

    if plot_features:
        plots_dir = out_dir / "plots"
        plot_selected_features(severity_df, merged, plot_features, plots_dir)
        print(f"Generated plots for features: {', '.join(plot_features)}")
        print(f"Plot files saved under: {plots_dir}")

    heatmap_dir = out_dir / "plots"
    plot_feature_heatmap(severity_df, heatmap_features, heatmap_dir)
    print(f"Combined feature heatmap saved to: {heatmap_dir / 'feature_heatmap.png'}")

    # 也打印一些全局统计
    print("=== Trend Consistency ===")
    print(trends["consistent"].value_counts(dropna=False))
    print("\n=== Ordinal Fit (pseudo R^2) ===")
    print(ordinal_summary["ordinal_pseudo_r2"].describe())
    print("\n=== Significant features (combined criteria) ===")
    print(f"{len(significant)} rows written to significant_features.csv")


if __name__ == "__main__":
    main()