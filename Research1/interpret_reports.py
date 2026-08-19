from pathlib import Path

import matplotlib.pyplot as plt

from para import PATH
from utils import ensure_dir


class InterpretationReporter:
    def summarize(self, rows):
        grouped = {}
        for row in rows:
            key = (row["model_name"], row["layer"])
            grouped.setdefault(key, []).append(row)
        summary = []
        for (model_name, layer), items in grouped.items():
            quality = [row["quality_score"] for row in items]
            interp = [row["interpretability_score"] for row in items]
            risk = [row["false_positive_risk"] for row in items]
            types = {}
            for row in items:
                types[row["feature_type"]] = types.get(row["feature_type"], 0) + 1
            total = max(len(items), 1)
            summary.append(
                {
                    "model_name": model_name,
                    "layer": layer,
                    "num_features": len(items),
                    "mean_quality_score": sum(quality) / len(quality),
                    "mean_interpretability_score": sum(interp) / len(interp),
                    "mean_false_positive_risk": sum(risk) / len(risk),
                    "undiscernible_ratio": types.get("undiscernible", 0) / total,
                    "mixed_explanation_ratio": types.get("mixed", 0) / total,
                    "content_ratio": types.get("content", 0) / total,
                    "position_ratio": types.get("position", 0) / total,
                    "low_level_ratio": types.get("low-level", 0) / total,
                }
            )
        return summary

    def plot_quality_summary(self, summary):
        if not summary:
            return []
        paths = []
        labels = [f"{row['model_name']}\nL{row['layer']}" for row in summary]
        quality = [row["mean_quality_score"] for row in summary]
        path = Path(PATH.figure_dir) / "phase6_mean_quality_score.png"
        ensure_dir(path.parent)
        plt.figure(figsize=(max(6, len(labels) * 0.7), 4))
        plt.bar(range(len(labels)), quality)
        plt.xticks(range(len(labels)), labels, rotation=45, ha="right")
        plt.ylabel("mean quality score")
        plt.tight_layout()
        plt.savefig(path, dpi=160)
        plt.close()
        paths.append(str(path))

        type_keys = [
            "content_ratio",
            "position_ratio",
            "mixed_explanation_ratio",
            "low_level_ratio",
            "undiscernible_ratio",
        ]
        bottoms = [0.0 for _ in summary]
        path = Path(PATH.figure_dir) / "phase6_feature_type_distribution.png"
        plt.figure(figsize=(max(6, len(labels) * 0.7), 4))
        for key in type_keys:
            vals = [row.get(key, 0.0) for row in summary]
            plt.bar(range(len(labels)), vals, bottom=bottoms, label=key)
            bottoms = [base + val for base, val in zip(bottoms, vals)]
        plt.xticks(range(len(labels)), labels, rotation=45, ha="right")
        plt.ylabel("ratio")
        plt.legend(fontsize=8)
        plt.tight_layout()
        plt.savefig(path, dpi=160)
        plt.close()
        paths.append(str(path))
        return paths

    def case_markdown(self, rows, items_by_id):
        lines = ["# Phase 6 Feature Interpretation Cases", ""]
        sorted_rows = sorted(rows, key=lambda row: row["quality_score"], reverse=True)
        cases = sorted_rows[:5] + sorted_rows[-5:]
        for row in cases:
            item = items_by_id[row["blinded_feature_id"]]
            if item is None:
                lines.extend(
                    [
                        f"## {row['blinded_feature_id']}",
                        "",
                        f"- Model: {row['model_name']}",
                        f"- Layer: {row['layer']}",
                        f"- Feature: {row['feature']}",
                        f"- Feature type: {row['feature_type']}",
                        f"- Quality score: {float(row['quality_score']):.3f}",
                        "",
                        "Explanation:",
                        "",
                        row["short_explanation"],
                        "",
                        "Top activating contexts unavailable from resumed score table.",
                        "",
                    ]
                )
                continue
            lines.extend(
                [
                    f"## {row['blinded_feature_id']}",
                    "",
                    f"- Model: {row['model_name']}",
                    f"- Layer: {row['layer']}",
                    f"- Feature: {row['feature']}",
                    f"- Feature type: {row['feature_type']}",
                    f"- Quality score: {row['quality_score']:.3f}",
                    f"- Interpretability score: {row['interpretability_score']}",
                    f"- Specificity score: {row['specificity_score']}",
                    f"- Coverage score: {row['coverage_score']}",
                    f"- False-positive risk: {row['false_positive_risk']}",
                    "",
                    "Explanation:",
                    "",
                    row["short_explanation"],
                    "",
                    "Top activating contexts:",
                    "",
                ]
            )
            for idx, ctx in enumerate(item["contexts"][:5]):
                lines.extend(
                    [
                        f"{idx + 1}. `{ctx['context']}`",
                        f"   activated token: `{ctx['activated_token']}`; activation: {ctx['activation']:.4f}; position: {ctx['position']}",
                        "",
                    ]
                )
        return "\n".join(lines)
