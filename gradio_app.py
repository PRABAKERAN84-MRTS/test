import time
from dataclasses import dataclass
from typing import Dict, List, Tuple

import gradio as gr
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.inspection import permutation_importance
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score, roc_curve
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler


@dataclass
class EnergyConfig:
    edge_power_w: float = 2.0
    cloud_power_w: float = 12.0
    network_cost_j: float = 0.8


def load_open_dataset() -> Tuple[pd.DataFrame, pd.Series, List[str]]:
    ds = load_breast_cancer(as_frame=True)
    x = ds.data
    y = ds.target
    labels = ["High Risk (malignant)", "Low Risk (benign)"]
    return x, y, labels


def train_stratified_mlp(
    x: pd.DataFrame,
    y: pd.Series,
    hidden_layers: Tuple[int, ...],
    lr: float,
    max_iter: int,
    folds: int,
    random_state: int,
) -> Dict[str, object]:
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, stratify=y, random_state=random_state
    )

    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_test_scaled = scaler.transform(x_test)

    cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=random_state)
    fold_scores = []

    for tr_idx, val_idx in cv.split(x_train_scaled, y_train):
        x_tr, x_val = x_train_scaled[tr_idx], x_train_scaled[val_idx]
        y_tr, y_val = y_train.iloc[tr_idx], y_train.iloc[val_idx]

        clf = MLPClassifier(
            hidden_layer_sizes=hidden_layers,
            learning_rate_init=lr,
            max_iter=max_iter,
            random_state=random_state,
            early_stopping=True,
            n_iter_no_change=15,
        )
        clf.fit(x_tr, y_tr)
        y_val_prob = clf.predict_proba(x_val)[:, 1]
        fold_scores.append(roc_auc_score(y_val, y_val_prob))

    final_model = MLPClassifier(
        hidden_layer_sizes=hidden_layers,
        learning_rate_init=lr,
        max_iter=max_iter,
        random_state=random_state,
        early_stopping=True,
        n_iter_no_change=15,
    )

    start_train = time.perf_counter()
    final_model.fit(x_train_scaled, y_train)
    train_seconds = time.perf_counter() - start_train

    start_infer = time.perf_counter()
    y_prob = final_model.predict_proba(x_test_scaled)[:, 1]
    y_pred = (y_prob >= 0.5).astype(int)
    infer_seconds = time.perf_counter() - start_infer

    return {
        "model": final_model,
        "x_test": x_test,
        "x_test_scaled": x_test_scaled,
        "y_test": y_test,
        "y_pred": y_pred,
        "y_prob": y_prob,
        "cv_auc_mean": float(np.mean(fold_scores)),
        "cv_auc_std": float(np.std(fold_scores)),
        "test_auc": float(roc_auc_score(y_test, y_prob)),
        "test_acc": float(accuracy_score(y_test, y_pred)),
        "classification_report": classification_report(
            y_test, y_pred, output_dict=True, zero_division=0
        ),
        "train_seconds": train_seconds,
        "infer_seconds": infer_seconds,
    }


def estimate_energy(probs: np.ndarray, threshold_edge: float, config: EnergyConfig) -> Dict[str, float]:
    confidence = np.abs(probs - 0.5) * 2
    edge_handled = confidence >= threshold_edge
    cloud_handled = ~edge_handled

    edge_count = int(edge_handled.sum())
    cloud_count = int(cloud_handled.sum())

    edge_energy = edge_count * config.edge_power_w
    cloud_energy = cloud_count * (config.cloud_power_w + config.network_cost_j)
    total_energy = edge_energy + cloud_energy

    return {
        "edge_count": edge_count,
        "cloud_count": cloud_count,
        "edge_energy": edge_energy,
        "cloud_energy": cloud_energy,
        "total_energy": total_energy,
    }


def explain_model(model, x_test_scaled, y_test, feature_names):
    perm = permutation_importance(
        model,
        x_test_scaled,
        y_test,
        scoring="roc_auc",
        n_repeats=8,
        random_state=7,
    )
    return pd.DataFrame(
        {"feature": feature_names, "importance": perm.importances_mean}
    ).sort_values("importance", ascending=False)


def local_explanation(sample_scaled: np.ndarray, feature_names: List[str]) -> pd.DataFrame:
    df = pd.DataFrame(
        {"feature": feature_names, "scaled_magnitude": np.abs(sample_scaled)}
    )
    return df.sort_values("scaled_magnitude", ascending=False).head(10)


def render_roc_curve(y_test: pd.Series, y_prob: np.ndarray):
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(fpr, tpr, label="ROC")
    ax.plot([0, 1], [0, 1], "--", color="gray", label="Random")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("Risk Prediction ROC Curve")
    ax.legend()
    fig.tight_layout()
    return fig


def render_energy_chart(energy: Dict[str, float]):
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.bar(["Edge", "Cloud"], [energy["edge_energy"], energy["cloud_energy"]])
    ax.set_ylabel("Energy (J)")
    ax.set_title("Energy-Aware Resource Allocation")
    fig.tight_layout()
    return fig


def run_pipeline(
    hidden_1,
    hidden_2,
    learning_rate,
    max_iter,
    folds,
    random_state,
    threshold,
    edge_power,
    cloud_power,
    network_cost,
    sample_idx,
):
    x, y, labels = load_open_dataset()
    hidden_layers = (int(hidden_1),) if int(hidden_2) == 0 else (int(hidden_1), int(hidden_2))

    metrics = train_stratified_mlp(
        x,
        y,
        hidden_layers=hidden_layers,
        lr=float(learning_rate),
        max_iter=int(max_iter),
        folds=int(folds),
        random_state=int(random_state),
    )

    energy = estimate_energy(
        metrics["y_prob"],
        float(threshold),
        EnergyConfig(
            edge_power_w=float(edge_power),
            cloud_power_w=float(cloud_power),
            network_cost_j=float(network_cost),
        ),
    )

    global_imp = explain_model(
        metrics["model"], metrics["x_test_scaled"], metrics["y_test"], x.columns.tolist()
    ).head(15)

    idx = max(0, min(int(sample_idx), len(metrics["x_test"]) - 1))
    local_df = local_explanation(metrics["x_test_scaled"][idx], x.columns.tolist())

    class_name = labels[int(metrics["y_pred"][idx])]
    pred_prob = float(metrics["y_prob"][idx])

    summary = (
        f"Test ROC-AUC: {metrics['test_auc']:.3f}\n"
        f"Test Accuracy: {metrics['test_acc']:.3f}\n"
        f"CV AUC: {metrics['cv_auc_mean']:.3f} ± {metrics['cv_auc_std']:.3f}\n"
        f"Training time: {metrics['train_seconds']:.3f}s | Inference time: {metrics['infer_seconds']:.5f}s\n\n"
        f"Edge cases: {energy['edge_count']} | Cloud cases: {energy['cloud_count']}\n"
        f"Edge Energy: {energy['edge_energy']:.1f} J | Cloud Energy: {energy['cloud_energy']:.1f} J\n"
        f"Total Energy: {energy['total_energy']:.1f} J\n\n"
        f"Selected patient prediction: {class_name} (low-risk prob={pred_prob:.3f})"
    )

    roc_fig = render_roc_curve(metrics["y_test"], metrics["y_prob"])
    energy_fig = render_energy_chart(energy)
    report_df = pd.DataFrame(metrics["classification_report"]).T.reset_index().rename(columns={"index": "class"})

    return summary, roc_fig, energy_fig, global_imp, local_df, report_df


def build_demo():
    with gr.Blocks(title="IoT Healthcare Risk & Energy Optimizer (Gradio)") as demo:
        gr.Markdown(
            "# Optimizing Cloud Resource Allocation for Stratified MLP IoT Healthcare\n"
            "Explainable risk prediction and energy-aware operations (Colab-ready Gradio app)."
        )

        with gr.Row():
            hidden_1 = gr.Slider(8, 128, value=32, step=8, label="Hidden Layer 1")
            hidden_2 = gr.Slider(0, 128, value=16, step=8, label="Hidden Layer 2 (0 = disabled)")
            learning_rate = gr.Dropdown([0.0005, 0.001, 0.003, 0.005, 0.01], value=0.001, label="Learning Rate")

        with gr.Row():
            max_iter = gr.Slider(100, 1000, value=350, step=50, label="Max Iterations")
            folds = gr.Slider(3, 8, value=5, step=1, label="Stratified CV folds")
            random_state = gr.Number(value=42, label="Random Seed", precision=0)

        with gr.Row():
            threshold = gr.Slider(0.1, 0.95, value=0.6, step=0.05, label="Edge confidence threshold")
            edge_power = gr.Number(value=2.0, label="Edge power per case (J)")
            cloud_power = gr.Number(value=12.0, label="Cloud power per case (J)")
            network_cost = gr.Number(value=0.8, label="Network overhead (J)")

        sample_idx = gr.Slider(0, 113, value=0, step=1, label="Sample index in test set")
        run_btn = gr.Button("Train & Evaluate")

        summary = gr.Textbox(label="Results Summary", lines=12)
        with gr.Row():
            roc_plot = gr.Plot(label="Risk Prediction ROC")
            energy_plot = gr.Plot(label="Energy Allocation")

        global_df = gr.Dataframe(label="Global Explainability (Permutation Importance)")
        local_df = gr.Dataframe(label="Local Explainability (Selected Patient)")
        report_df = gr.Dataframe(label="Classification Report")

        run_btn.click(
            fn=run_pipeline,
            inputs=[
                hidden_1,
                hidden_2,
                learning_rate,
                max_iter,
                folds,
                random_state,
                threshold,
                edge_power,
                cloud_power,
                network_cost,
                sample_idx,
            ],
            outputs=[summary, roc_plot, energy_plot, global_df, local_df, report_df],
        )
    return demo


def launch_colab_app(share: bool = True):
    demo = build_demo()
    demo.launch(share=share, inline=False)


if __name__ == "__main__":
    build_demo().launch()
