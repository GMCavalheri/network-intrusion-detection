import logging
import os
from logging.handlers import RotatingFileHandler

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import streamlit as st

API_BASE_URL = os.environ.get("API_BASE_URL", "http://localhost:8000")
LOG_DIR = os.environ.get("LOG_DIR", "/opt/logs")
DATASET_LABELS = {"nsl_kdd": "NSL-KDD", "cicids2017": "CICIDS2017"}

st.set_page_config(page_title="Network Intrusion Detection Dashboard", page_icon="🛰️", layout="wide")


def _get_logger() -> logging.Logger:
    logger = logging.getLogger("frontend")
    if logger.handlers:
        return logger
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s | %(levelname)-7s | %(name)s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S")

    console = logging.StreamHandler()
    console.setFormatter(fmt)
    logger.addHandler(console)

    try:
        os.makedirs(LOG_DIR, exist_ok=True)
        file_handler = RotatingFileHandler(os.path.join(LOG_DIR, "frontend.log"), maxBytes=5_000_000, backupCount=3)
        file_handler.setFormatter(fmt)
        logger.addHandler(file_handler)
    except OSError:
        pass

    logger.propagate = False
    return logger


logger = _get_logger()


@st.cache_data(ttl=30)
def api_get(path, params=None):
    try:
        r = requests.get(f"{API_BASE_URL}{path}", params=params, timeout=30)
        r.raise_for_status()
        return r.json()
    except requests.RequestException:
        logger.exception("GET %s%s failed", API_BASE_URL, path)
        raise


def render_overview(dataset):
    st.title(f"Overview — {DATASET_LABELS[dataset]}")
    try:
        summary = api_get("/stats/summary", {"dataset": dataset})
    except requests.HTTPError:
        st.warning("No scored data yet. Run the Spark pipeline first: `docker compose run --rm spark-pipeline`.")
        return

    total = summary["total_flows"]
    attack_rate = summary["predicted_attack_count"] / total if total else 0
    metrics = summary.get("model_metrics") or {}
    auc = metrics.get("auc_roc")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Flows", f"{total:,}")
    c2.metric("Predicted Attack Rate", f"{attack_rate:.2%}")
    c3.metric("Avg. Attack Probability", f"{summary['avg_attack_probability']:.3f}" if summary.get("avg_attack_probability") is not None else "n/a")
    c4.metric("Model AUC-ROC", f"{auc:.3f}" if auc else "n/a")

    col_a, col_b = st.columns(2)
    with col_a:
        cats = pd.DataFrame(api_get("/stats/categories", {"dataset": dataset}))
        if not cats.empty:
            fig = px.bar(cats, x="attack_category", y="predicted_attack_count", title="Predicted Attacks by Category")
            fig.update_xaxes(title="")
            fig.update_yaxes(title="Predicted attack count")
            st.plotly_chart(fig, use_container_width=True)
    with col_b:
        ds_stats = pd.DataFrame(api_get("/stats/datasets"))
        if not ds_stats.empty:
            mine = ds_stats[ds_stats["dataset_source"] == dataset]
            if not mine.empty:
                fig2 = px.bar(mine, x="split", y=["total_flows", "actual_attack_count", "predicted_attack_count"],
                              barmode="group", title="Train vs. Test Split")
                fig2.update_xaxes(title="")
                st.plotly_chart(fig2, use_container_width=True)


def render_data_quality(dataset):
    st.title(f"Data Quality Report — {DATASET_LABELS[dataset]}")
    st.caption("What the Spark ETL step found (and, where possible, fixed) in the raw data.")
    try:
        report = api_get("/reports/data-quality", {"dataset": dataset})
    except requests.HTTPError:
        st.warning("Report not available yet - run the Spark pipeline first.")
        return

    if dataset == "nsl_kdd":
        c1, c2, c3 = st.columns(3)
        c1.metric("Total raw rows", f"{report['total_raw_rows']:,}")
        c2.metric("Duplicates removed", f"{report['duplicates_removed']:,}")
        c3.metric("Final cleaned rows", f"{report['final_cleaned_rows']:,}")
        st.info(
            f"**{len(report['novel_attack_types_in_test_not_in_train'])} attack types** appear in the test set "
            "that never occur in training - NSL-KDD is deliberately designed this way, to test generalization "
            "to unseen attacks rather than memorization. See Model Performance for the detection rate on exactly these rows."
        )
        with st.expander("Novel attack types (test-only)"):
            st.write(", ".join(report["novel_attack_types_in_test_not_in_train"]))
    else:
        c1, c2, c3 = st.columns(3)
        c1.metric("Total raw rows", f"{report['total_raw_rows']:,}")
        c2.metric("Duplicates removed", f"{report['duplicates_removed']:,}")
        c3.metric("Final cleaned rows", f"{report['final_cleaned_rows']:,}")
        c4, c5 = st.columns(2)
        c4.metric("Negative-duration rows dropped", report["negative_duration_rows_dropped"])
        inf_nan = sum(report["rate_column_infinity_or_nan_imputed"].values())
        c5.metric("Infinity/NaN rate values imputed", inf_nan)
        st.caption("Rate columns (bytes/s, packets/s) are literally 'Infinity'/'NaN' in the raw CSV when a flow's "
                   "duration is 0 - a division-by-zero artifact of CICFlowMeter, imputed to a -1 sentinel.")

    with st.expander("Full report JSON"):
        st.json(report)


def render_model_performance(dataset):
    st.title(f"Model Performance — {DATASET_LABELS[dataset]}")
    try:
        summary = api_get("/stats/summary", {"dataset": dataset})
    except requests.HTTPError:
        st.warning("No metrics yet - run the Spark pipeline first.")
        return

    metrics = summary.get("model_metrics")
    if not metrics:
        st.warning("Model metrics not found.")
        return

    st.caption(f"{metrics['model_type']} trained on {metrics['train_rows']:,} rows, tested on {metrics['test_rows']:,} rows. "
               f"Split strategy: {metrics.get('split_strategy', 'n/a')}.")

    st.subheader("At the default threshold (0.5)")
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("AUC-ROC", f"{metrics['auc_roc']:.3f}")
    c2.metric("AUC-PR", f"{metrics['auc_pr']:.3f}")
    c3.metric("Precision", f"{metrics['precision']:.3f}")
    c4.metric("Recall", f"{metrics['recall']:.3f}")
    c5.metric("F1", f"{metrics['f1']:.3f}")

    best = metrics.get("best_threshold")
    if best and best["threshold"] != 0.5:
        st.subheader(f"At the empirically best threshold ({best['threshold']})")
        st.caption(
            "A high AUC-ROC only means the model *ranks* attacks above benign traffic on average - it says "
            "nothing about whether 0.5 is a good cutoff. Tuning the threshold to the actual precision/recall "
            "tradeoff a security team wants (their false-positive budget) can look completely different."
        )
        b1, b2, b3 = st.columns(3)
        b1.metric("Precision", f"{best['precision']:.3f}", delta=round(best["precision"] - metrics["precision"], 3))
        b2.metric("Recall", f"{best['recall']:.3f}", delta=round(best["recall"] - metrics["recall"], 3))
        b3.metric("F1", f"{best['f1']:.3f}", delta=round(best["f1"] - metrics["f1"], 3))

    curve = pd.DataFrame(metrics.get("threshold_curve", []))
    if not curve.empty:
        curve = curve.sort_values("threshold")
        fig_curve = px.line(curve, x="threshold", y=["precision", "recall", "f1"],
                             title="Precision / Recall / F1 vs. Classification Threshold")
        fig_curve.update_yaxes(title="Score", range=[0, 1])
        st.plotly_chart(fig_curve, use_container_width=True)

    if "novel_attack_detection_rate" in metrics:
        st.metric(
            "Detection rate on attack types never seen in training",
            f"{metrics['novel_attack_detection_rate']:.1%}",
            help=f"{metrics['novel_caught']:,} of {metrics['novel_total']:,} rows whose attack_type has zero examples in KDDTrain+.",
        )

    col_a, col_b = st.columns(2)
    with col_a:
        cm = metrics["confusion_matrix"]
        z = [[cm["true_negative"], cm["false_positive"]], [cm["false_negative"], cm["true_positive"]]]
        fig_cm = go.Figure(data=go.Heatmap(
            z=z, x=["Predicted Benign", "Predicted Attack"], y=["Actual Benign", "Actual Attack"],
            text=z, texttemplate="%{text}", colorscale="Blues",
        ))
        fig_cm.update_layout(title="Confusion Matrix (test set, @0.5 threshold)")
        st.plotly_chart(fig_cm, use_container_width=True)
    with col_b:
        fi = pd.DataFrame(metrics["feature_importances"][:15], columns=["feature", "importance"]).sort_values("importance")
        fig_fi = px.bar(fi, x="importance", y="feature", orientation="h", title="Top 15 Feature Importances")
        st.plotly_chart(fig_fi, use_container_width=True)

    cb = pd.DataFrame(metrics.get("category_breakdown", []))
    if not cb.empty:
        st.subheader("Detection rate by attack category (test set)")
        st.dataframe(cb, use_container_width=True, hide_index=True)


def render_flow_explorer():
    st.title("Flow Explorer")
    meta = api_get("/meta")

    with st.form("filters"):
        c1, c2, c3 = st.columns(3)
        dataset = c1.selectbox("Dataset", ["All"] + meta["datasets"], format_func=lambda d: DATASET_LABELS.get(d, d))
        category = c2.selectbox("Attack category", ["All"] + meta["attack_categories"])
        label = c3.selectbox("Predicted", ["All", "Attack", "Benign"])
        st.form_submit_button("Search")

    params = {"limit": 200}
    if dataset != "All":
        params["dataset"] = dataset
    if category != "All":
        params["attack_category"] = category
    if label == "Attack":
        params["predicted_label"] = 1
    elif label == "Benign":
        params["predicted_label"] = 0

    data = api_get("/flows", params)
    st.caption(f"{data['total']:,} matching flows (showing first {len(data['items'])})")
    df = pd.DataFrame(data["items"])
    if not df.empty:
        df["attack?"] = df["predicted_label"].map({1: "🚨 Attack", 0: "Benign"})
        st.dataframe(
            df[["flow_id", "dataset_source", "split", "source_day", "protocol", "duration",
                "bytes_sent", "bytes_received", "attack_category_actual", "attack?",
                "attack_probability", "rule_flags"]],
            use_container_width=True, hide_index=True,
        )
        with st.expander("Raw feature vector for a flow"):
            flow_id = st.selectbox("Flow ID", df["flow_id"])
            st.json(df[df["flow_id"] == flow_id]["raw_features"].iloc[0])
    else:
        st.info("No flows match these filters.")


def render_live_demo():
    st.title("Live Intrusion Detection Demo")
    st.caption("Scores a hypothetical network flow against the exact Spark MLlib model trained by the pipeline.")
    meta = api_get("/meta")

    dataset = st.selectbox("Dataset", meta["datasets"], format_func=lambda d: DATASET_LABELS.get(d, d))
    form_fields = meta[dataset]["form_fields"]

    with st.form("score_form"):
        values = {}
        cols = st.columns(3)
        for i, field in enumerate(form_fields):
            col = cols[i % 3]
            if dataset == "nsl_kdd" and field == "protocol_type":
                values[field] = col.selectbox(field, meta["nsl_kdd"]["protocol_types"])
            elif dataset == "nsl_kdd" and field == "service":
                values[field] = col.selectbox(field, meta["nsl_kdd"]["services"])
            elif dataset == "nsl_kdd" and field == "flag":
                values[field] = col.selectbox(field, meta["nsl_kdd"]["flags"])
            else:
                values[field] = col.number_input(field, value=0.0, step=1.0)
        submitted = st.form_submit_button("Score Flow", type="primary")

    if not submitted:
        return

    payload = {"dataset": dataset, "features": values}
    with st.spinner("Scoring with the Spark MLlib model..."):
        resp = requests.post(f"{API_BASE_URL}/score", json=payload, timeout=60)

    if resp.status_code != 200:
        logger.error("POST /score failed with %d: %s", resp.status_code, resp.text)
        st.error(f"Scoring failed: {resp.text}")
        return

    result = resp.json()
    prob = result["attack_probability"]
    logger.info("Live demo score: dataset=%s probability=%.4f label=%d", dataset, prob, result["predicted_label"])

    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=prob * 100, number={"suffix": "%"}, title={"text": "Attack Probability"},
        gauge={
            "axis": {"range": [0, 100]},
            "bar": {"color": "#b3261e" if prob >= 0.5 else "#1e7a3d"},
            "steps": [{"range": [0, 50], "color": "#d9f2df"}, {"range": [50, 100], "color": "#fbdada"}],
        },
    ))
    fig.update_layout(height=320, margin=dict(l=20, r=20, t=50, b=10))

    col_g, col_r = st.columns([1, 1])
    with col_g:
        st.plotly_chart(fig, use_container_width=True)
    with col_r:
        st.subheader("🚨 FLAGGED AS ATTACK" if result["predicted_label"] == 1 else "✅ Looks benign")
        st.caption("Default 0.5 threshold shown above - see Model Performance for why a tuned threshold may differ.")
        st.write("**Rule-based flags:** " + (", ".join(result["rule_flags"]) or "none"))
        with st.expander("Full feature vector fed to the model"):
            st.json(result["features_used"])


def main():
    st.sidebar.title("🛰️ Network Intrusion Detection")
    st.sidebar.caption("NSL-KDD + CICIDS2017 → Spark ETL/MLlib → FastAPI → Streamlit")
    page = st.sidebar.radio(
        "Section",
        ["Overview", "Data Quality Report", "Model Performance", "Flow Explorer", "Live Detection Demo"],
    )

    dataset = None
    if page in ("Overview", "Data Quality Report", "Model Performance"):
        dataset = st.sidebar.selectbox("Dataset", ["nsl_kdd", "cicids2017"], format_func=lambda d: DATASET_LABELS.get(d, d))

    st.sidebar.divider()
    st.sidebar.markdown("[View source on GitHub](https://github.com/GMCavalheri/network-intrusion-detection)")

    try:
        api_get("/health")
    except requests.RequestException as e:
        st.error(f"Cannot reach the API at {API_BASE_URL}: {e}")
        st.stop()

    if page == "Overview":
        render_overview(dataset)
    elif page == "Data Quality Report":
        render_data_quality(dataset)
    elif page == "Model Performance":
        render_model_performance(dataset)
    elif page == "Flow Explorer":
        render_flow_explorer()
    elif page == "Live Detection Demo":
        render_live_demo()


if __name__ == "__main__":
    main()
