import time
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import streamlit as st
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
    """Use an open-source healthcare dataset bundled in scikit-learn."""
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
        x,
        y,
        test_size=0.2,
        stratify=y,
        random_state=random_state,
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

    result = {
        "model": final_model,
        "scaler": scaler,
        "x_test": x_test,
        "x_test_scaled": x_test_scaled,
        "y_test": y_test,
        "y_pred": y_pred,
        "y_prob": y_prob,
        "cv_auc_mean": float(np.mean(fold_scores)),
        "cv_auc_std": float(np.std(fold_scores)),
        "test_auc": float(roc_auc_score(y_test, y_prob)),
        "test_acc": float(accuracy_score(y_test, y_pred)),
        "classification_report": classification_report(y_test, y_pred, output_dict=True, zero_division=0),
        "train_seconds": train_seconds,
        "infer_seconds": infer_seconds,
    }
    return result


def estimate_energy(
    probs: np.ndarray,
    threshold_edge: float,
    config: EnergyConfig,
) -> Dict[str, float]:
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
    imp_df = pd.DataFrame(
        {
            "feature": feature_names,
            "importance": perm.importances_mean,
        }
    ).sort_values("importance", ascending=False)
    return imp_df


def local_explanation(sample_scaled: np.ndarray, feature_names: List[str]) -> pd.DataFrame:
    contributions = np.abs(sample_scaled)
    df = pd.DataFrame({"feature": feature_names, "scaled_magnitude": contributions})
    return df.sort_values("scaled_magnitude", ascending=False).head(10)


def main():
    st.set_page_config(page_title="IoT Healthcare Risk & Energy Optimizer", layout="wide")
    st.title("Optimizing Cloud Resource Allocation for Stratified MLP IoT Healthcare")
    st.caption(
        "Explainable risk prediction and energy-aware operations using open-source data"
    )

    with st.sidebar:
        st.header("Model Settings")
        hidden_1 = st.slider("Hidden Layer 1", 8, 128, 32, step=8)
        hidden_2 = st.slider("Hidden Layer 2", 0, 128, 16, step=8)
        learning_rate = st.select_slider(
            "Learning Rate", options=[0.0005, 0.001, 0.003, 0.005, 0.01], value=0.001
        )
        max_iter = st.slider("Max Iterations", 100, 1000, 350, step=50)
        folds = st.slider("Stratified CV folds", 3, 8, 5)
        random_state = st.number_input("Random Seed", min_value=0, max_value=9999, value=42)

        st.header("Energy Settings")
        threshold = st.slider("Edge confidence threshold", 0.1, 0.95, 0.6, step=0.05)
        edge_power = st.number_input("Edge power per case (J)", min_value=0.1, value=2.0)
        cloud_power = st.number_input("Cloud power per case (J)", min_value=0.1, value=12.0)
        network_cost = st.number_input("Network overhead (J)", min_value=0.0, value=0.8)

        run_btn = st.button("Train & Evaluate")

    if run_btn:
        x, y, labels = load_open_dataset()
        hidden_layers = (hidden_1,) if hidden_2 == 0 else (hidden_1, hidden_2)

        with st.spinner("Training stratified MLP and computing explainability..."):
            metrics = train_stratified_mlp(
                x,
                y,
                hidden_layers=hidden_layers,
                lr=float(learning_rate),
                max_iter=max_iter,
                folds=folds,
                random_state=int(random_state),
            )

            energy = estimate_energy(
                metrics["y_prob"],
                threshold,
                EnergyConfig(
                    edge_power_w=float(edge_power),
                    cloud_power_w=float(cloud_power),
                    network_cost_j=float(network_cost),
                ),
            )

            global_imp = explain_model(
                metrics["model"],
                metrics["x_test_scaled"],
                metrics["y_test"],
                x.columns.tolist(),
            )

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Test ROC-AUC", f"{metrics['test_auc']:.3f}")
        c2.metric("Test Accuracy", f"{metrics['test_acc']:.3f}")
        c3.metric("CV AUC (mean±std)", f"{metrics['cv_auc_mean']:.3f} ± {metrics['cv_auc_std']:.3f}")
        c4.metric("Total Energy (J)", f"{energy['total_energy']:.1f}")

        st.subheader("Risk Prediction Curve")
        fpr, tpr, _ = roc_curve(metrics["y_test"], metrics["y_prob"])
        roc_df = pd.DataFrame({"FPR": fpr, "TPR": tpr})
        st.line_chart(roc_df.set_index("FPR"))

        st.subheader("Energy-Aware Resource Allocation")
        e1, e2, e3, e4 = st.columns(4)
        e1.metric("Edge cases", energy["edge_count"])
        e2.metric("Cloud cases", energy["cloud_count"])
        e3.metric("Edge Energy (J)", f"{energy['edge_energy']:.1f}")
        e4.metric("Cloud Energy (J)", f"{energy['cloud_energy']:.1f}")

        energy_df = pd.DataFrame(
            {
                "Layer": ["Edge", "Cloud"],
                "Energy": [energy["edge_energy"], energy["cloud_energy"]],
            }
        )
        st.bar_chart(energy_df.set_index("Layer"))

        st.subheader("Global Explainability (Permutation Importance)")
        st.dataframe(global_imp.head(15), use_container_width=True)

        st.subheader("Local Explainability (Top Drivers for One Patient)")
        sample_idx = st.slider("Sample index in test set", 0, len(metrics["x_test"]) - 1, 0)
        sample_scaled = metrics["x_test_scaled"][sample_idx]
        local_df = local_explanation(sample_scaled, x.columns.tolist())
        st.dataframe(local_df, use_container_width=True)

        class_name = labels[int(metrics["y_pred"][sample_idx])]
        pred_prob = float(metrics["y_prob"][sample_idx])
        st.info(
            f"Predicted class for selected patient: **{class_name}** with probability {pred_prob:.3f} for low-risk class"
        )

        st.subheader("Classification Report")
        st.json(metrics["classification_report"])

        st.caption(
            f"Training time: {metrics['train_seconds']:.3f}s | Inference time: {metrics['infer_seconds']:.5f}s"
        )
    else:
        st.write(
            "Configure parameters from the sidebar and click **Train & Evaluate** to run the full pipeline."
        )


if __name__ == "__main__":
    main()
