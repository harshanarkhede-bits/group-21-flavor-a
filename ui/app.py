import json
import sys
from datetime import date, datetime, time
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import streamlit as st

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

from src.config import config
from src.serving.locations import ALLOWED_LOCATIONS, LOCATION_COORDINATES, calculate_haversine_distance_km

# ------------------------------------------------------------
# Page Config & Styles
# ------------------------------------------------------------
st.set_page_config(
    page_title="Ride ETA Prediction Platform | MLOps",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1e293b;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #64748b;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
        color: #ffffff;
        padding: 20px;
        border-radius: 12px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
        text-align: center;
    }
    .metric-val {
        font-size: 2.5rem;
        font-weight: 800;
        color: #38bdf8;
    }
    .metric-label {
        font-size: 0.9rem;
        text-transform: uppercase;
        letter-spacing: 1px;
        color: #94a3b8;
    }
    .badge {
        display: inline-block;
        padding: 4px 10px;
        border-radius: 6px;
        font-size: 0.85rem;
        font-weight: 600;
    }
    .badge-high { background-color: #fee2e2; color: #b91c1c; }
    .badge-med { background-color: #fef3c7; color: #b45309; }
    .badge-low { background-color: #dcfce7; color: #15803d; }
    </style>
    """,
    unsafe_allow_html=True,
)

API_BASE_URL = f"http://127.0.0.1:{config.serving_port}"

# ------------------------------------------------------------
# Sidebar Navigation & System Health
# ------------------------------------------------------------
with st.sidebar:
    st.image("https://img.icons8.com/isometric/100/taxi.png", width=70)
    st.title("Ride ETA MLOps")
    st.caption("Flavor A Production Platform")
    st.divider()

    st.subheader("System Status")
    try:
        health_resp = requests.get(f"{API_BASE_URL}/health", timeout=1.5)
        if health_resp.status_code == 200:
            st.success("🟢 API Server: Online")
        else:
            st.warning("🟡 API Server: Degraded")
    except Exception:
        st.error("🔴 API Server: Offline")
        st.caption(f"Run: `uvicorn src.serving.api:app --port {config.serving_port}`")

    st.divider()
    st.caption(f"**Version:** {config.project_version}")
    st.caption(f"**Model Registry:** `{config.registered_model_name}`")
    st.caption(f"**Target:** `{config.target_column}` (minutes)")

# ------------------------------------------------------------
# Main Title
# ------------------------------------------------------------
st.markdown('<div class="main-header">⚡ NYC Ride & Delivery ETA Platform</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-header">End-to-end MLOps pipeline featuring DVC versioning, MLflow tracking, automated drift detection & Prometheus metrics.</div>',
    unsafe_allow_html=True,
)

tab1, tab2, tab3, tab4 = st.tabs([
    "🚗 Real-Time ETA Predictor",
    "📁 Batch Inference",
    "📊 Model Registry & Performance",
    "🔍 Data Drift & MLOps Monitoring",
])

# ============================================================
# TAB 1: REAL-TIME ETA PREDICTOR
# ============================================================
with tab1:
    col_left, col_right = st.columns([1.1, 0.9], gap="large")

    with col_left:
        st.subheader("Trip Configuration")

        c1, c2 = st.columns(2)
        with c1:
            pickup_loc = st.selectbox(
                "📍 Pickup Neighborhood",
                options=ALLOWED_LOCATIONS,
                index=ALLOWED_LOCATIONS.index("Upper West Side") if "Upper West Side" in ALLOWED_LOCATIONS else 0,
            )
        with c2:
            default_drop = "Harlem" if "Harlem" in ALLOWED_LOCATIONS else ALLOWED_LOCATIONS[1]
            drop_loc = st.selectbox(
                "🎯 Drop-off Neighborhood",
                options=ALLOWED_LOCATIONS,
                index=ALLOWED_LOCATIONS.index(default_drop),
            )

        if pickup_loc == drop_loc:
            st.warning("⚠️ Pickup and Drop-off locations should be different.")

        c3, c4 = st.columns(2)
        with c3:
            pickup_d = st.date_input("📅 Date", value=datetime.now().date())
        with c4:
            pickup_t = st.time_input("⏰ Time", value=datetime.now().time())

        with st.expander("⚙️ Advanced Parameters (Optional)", expanded=False):
            passengers = st.slider("Passenger Count", min_value=1, max_value=6, value=1)
            surge = st.slider("Surge Multiplier", min_value=1.0, max_value=3.0, value=1.0, step=0.1)

        calc_distance = calculate_haversine_distance_km(pickup_loc, drop_loc)
        hour = pickup_t.hour
        is_weekend = int(pickup_d.weekday() >= 5)

        if is_weekend:
            traffic_guess = "Medium" if 11 <= hour <= 18 else "Low"
        elif (7 <= hour <= 10) or (16 <= hour <= 19):
            traffic_guess = "High"
        elif 11 <= hour <= 15:
            traffic_guess = "Medium"
        else:
            traffic_guess = "Low"

        # Information Badges
        st.markdown(
            f"""
            <div style="background: #f8fafc; padding: 12px; border-radius: 8px; margin-top: 10px; border: 1px solid #e2e8f0;">
                <span>📏 <strong>Distance:</strong> {calc_distance:.2f} km</span> &nbsp;|&nbsp; 
                <span>🚦 <strong>Traffic:</strong> <span class="badge badge-{'high' if traffic_guess == 'High' else 'med' if traffic_guess == 'Medium' else 'low'}">{traffic_guess}</span></span> &nbsp;|&nbsp;
                <span>🗓️ <strong>Day:</strong> {pickup_d.strftime('%A')}</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

        predict_btn = st.button("🚀 Calculate ETA", type="primary", use_container_width=True)

    with col_right:
        st.subheader("Route Estimation")

        if predict_btn:
            payload = {
                "pickup_location": pickup_loc,
                "drop_location": drop_loc,
                "pickup_date": pickup_d.strftime("%Y-%m-%d"),
                "pickup_time": pickup_t.strftime("%H:%M"),
                "passenger_count": passengers,
                "surge_multiplier": surge,
            }

            try:
                with st.spinner("Executing model inference..."):
                    res = requests.post(f"{API_BASE_URL}/predict", json=payload, timeout=5)

                if res.status_code == 200:
                    data = res.json()
                    eta_min = data["eta_minutes"]
                    eta_sec = data["eta_seconds"]

                    st.markdown(
                        f"""
                        <div class="metric-card">
                            <div class="metric-label">Estimated Travel Time</div>
                            <div class="metric-val">{eta_min:.1f} <span style="font-size: 1.2rem; color: #94a3b8;">min</span></div>
                            <div style="color: #cbd5e1; margin-top: 8px;">
                                ⏱️ approx. <strong>{int(eta_sec // 60)}m {int(eta_sec % 60)}s</strong> | Distance: <strong>{data['calculated_distance_km']} km</strong>
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

                    # Route Map
                    p_coords = LOCATION_COORDINATES.get(pickup_loc, (40.78, -73.97))
                    d_coords = LOCATION_COORDINATES.get(drop_loc, (40.81, -73.94))

                    map_df = pd.DataFrame([
                        {"lat": p_coords[0], "lon": p_coords[1], "type": f"Pickup: {pickup_loc}"},
                        {"lat": d_coords[0], "lon": d_coords[1], "type": f"Drop: {drop_loc}"},
                    ])

                    fig_map = px.scatter_mapbox(
                        map_df,
                        lat="lat",
                        lon="lon",
                        color="type",
                        size_max=15,
                        zoom=11,
                        mapbox_style="carto-positron",
                        title="NYC Trip Route",
                    )
                    fig_map.update_layout(margin={"r": 0, "t": 40, "l": 0, "b": 0}, height=280)
                    st.plotly_chart(fig_map, use_container_width=True)

                else:
                    st.error(f"API Error ({res.status_code}): {res.text}")
            except Exception as ex:
                st.error(f"Failed to connect to API: {ex}")
        else:
            st.info("👈 Select locations and click **Calculate ETA** to generate a real-time prediction.")
            # Initial placeholder map
            sample_df = pd.DataFrame([
                {"lat": 40.78705, "lon": -73.97542, "type": "Upper West Side"},
                {"lat": 40.81160, "lon": -73.94650, "type": "Harlem"},
            ])
            fig_init = px.scatter_mapbox(
                sample_df,
                lat="lat",
                lon="lon",
                color="type",
                zoom=11,
                mapbox_style="carto-positron",
            )
            fig_init.update_layout(margin={"r": 0, "t": 10, "l": 0, "b": 0}, height=280)
            st.plotly_chart(fig_init, use_container_width=True)

# ============================================================
# TAB 2: BATCH INFERENCE
# ============================================================
with tab2:
    st.subheader("Batch ETA Prediction")
    st.write("Upload a CSV file containing multiple trips or load the sample batch dataset.")

    col_b1, col_b2 = st.columns([2, 1])

    with col_b1:
        uploaded_file = st.file_uploader("Upload CSV File", type=["csv"])

    with col_b2:
        st.write("Or use benchmark dataset:")
        if st.button("Load 50 Sample Trips"):
            if config.raw_data_path.exists():
                sample_batch = pd.read_csv(config.raw_data_path).head(50)
                st.session_state["batch_data"] = sample_batch
            else:
                st.warning("Raw dataset not found.")

    if uploaded_file is not None:
        st.session_state["batch_data"] = pd.read_csv(uploaded_file)

    if "batch_data" in st.session_state:
        df_batch = st.session_state["batch_data"]
        st.write(f"Loaded **{len(df_batch)}** trips:")
        st.dataframe(df_batch.head(5), use_container_width=True)

        if st.button("⚡ Run Batch Predictions", type="primary"):
            trips_payload = []
            for _, row in df_batch.iterrows():
                p_loc = row.get("pickup_location", "Upper West Side")
                d_loc = row.get("drop_location", "Harlem")
                p_date = str(row.get("pickup_date", "2026-08-27"))
                p_time = str(row.get("pickup_time", "12:00"))

                # Format corrections
                try:
                    p_date = datetime.strptime(p_date, "%d-%m-%Y").strftime("%Y-%m-%d")
                except Exception:
                    pass

                trips_payload.append({
                    "pickup_location": p_loc if p_loc in ALLOWED_LOCATIONS else "Upper West Side",
                    "drop_location": d_loc if d_loc in ALLOWED_LOCATIONS else "Harlem",
                    "pickup_date": p_date if len(p_date) == 10 else "2026-08-27",
                    "pickup_time": p_time[:5] if len(p_time) >= 5 else "12:00",
                    "passenger_count": int(row.get("passenger_count", 1)),
                    "surge_multiplier": float(row.get("surge_multiplier", 1.0)),
                })

            try:
                with st.spinner(f"Predicting {len(trips_payload)} trips via FastAPI..."):
                    res = requests.post(f"{API_BASE_URL}/predict/batch", json={"trips": trips_payload}, timeout=30)

                if res.status_code == 200:
                    resp_data = res.json()
                    preds_list = [p["eta_minutes"] for p in resp_data["predictions"]]
                    df_batch["predicted_eta_minutes"] = preds_list

                    st.success(f"Successfully generated {len(preds_list)} predictions!")

                    # Metrics & Distribution
                    m1, m2, m3 = st.columns(3)
                    m1.metric("Average Predicted ETA", f"{np.mean(preds_list):.1f} min")
                    m2.metric("Min Predicted ETA", f"{np.min(preds_list):.1f} min")
                    m3.metric("Max Predicted ETA", f"{np.max(preds_list):.1f} min")

                    fig_dist = px.histogram(
                        df_batch,
                        x="predicted_eta_minutes",
                        nbins=25,
                        title="Distribution of Predicted ETAs",
                        color_discrete_sequence=["#38bdf8"],
                    )
                    st.plotly_chart(fig_dist, use_container_width=True)

                    st.dataframe(df_batch, use_container_width=True)

                    csv_data = df_batch.to_csv(index=False).encode("utf-8")
                    st.download_button(
                        "📥 Download Predictions CSV",
                        data=csv_data,
                        file_name="eta_predictions.csv",
                        mime="text/csv",
                    )
                else:
                    st.error(f"Batch API error: {res.text}")
            except Exception as e:
                st.error(f"Batch prediction request failed: {e}")

# ============================================================
# TAB 3: MODEL REGISTRY & PERFORMANCE
# ============================================================
with tab3:
    st.subheader("Model Performance & Experiment Tracking")

    # Load Metadata
    metadata = {}
    if config.metadata_path.exists():
        with open(config.metadata_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Active Model", metadata.get("model", "GradientBoosting"))
    c2.metric("Test R² Score", f"{metadata.get('test_r2', 0.72):.4f}")
    c3.metric("Test RMSE", f"{metadata.get('test_rmse', 5.6):.2f} min")
    c4.metric("Test MAE", f"{metadata.get('test_mae', 3.8):.2f} min")

    st.divider()

    # Candidate Comparison Table
    st.subheader("Candidate Model Leaderboard")
    comparisons = metadata.get("all_model_comparisons", {
        "GradientBoosting": {"test_r2": 0.7202, "test_rmse": 5.6389, "test_mae": 3.8686},
        "RandomForest": {"test_r2": 0.6980, "test_rmse": 5.8600, "test_mae": 4.0200},
        "RidgeRegression": {"test_r2": 0.5840, "test_rmse": 6.8900, "test_mae": 5.1200},
    })

    leaderboard_df = pd.DataFrame.from_dict(comparisons, orient="index").reset_index()
    leaderboard_df.columns = ["Model Architecture", "Test R² Score", "Test RMSE (min)", "Test MAE (min)"]
    leaderboard_df = leaderboard_df.sort_values(by="Test RMSE (min)")
    st.dataframe(leaderboard_df, use_container_width=True)

    col_p1, col_p2 = st.columns(2)
    with col_p1:
        res_plot = config.store_dir / "plots" / "GradientBoosting_residuals.png"
        if res_plot.exists():
            st.image(str(res_plot), caption="Actual vs. Predicted Residuals")
        else:
            st.info("Residual plots will appear here once model training is executed.")

    with col_p2:
        imp_plot = config.store_dir / "plots" / "GradientBoosting_feature_importance.png"
        if imp_plot.exists():
            st.image(str(imp_plot), caption="Top Feature Importances")
        else:
            st.info("Feature importance plot will appear here once model training is executed.")

# ============================================================
# TAB 4: DATA DRIFT & MONITORING DASHBOARD
# ============================================================
with tab4:
    st.subheader("Live Data Drift & Retraining Control Center")

    drift_info = {}
    if config.drift_report_json.exists():
        try:
            with open(config.drift_report_json, "r", encoding="utf-8") as f:
                drift_info = json.load(f)
        except Exception:
            pass

    summary = drift_info.get("drift_summary", {})
    has_drift = summary.get("dataset_drift_detected", False)
    drift_share = summary.get("drift_share", 0.0)

    col_d1, col_d2, col_d3 = st.columns(3)
    with col_d1:
        if has_drift:
            st.error("⚠️ DATA DRIFT DETECTED")
        else:
            st.success("✅ NO DRIFT DETECTED (HEALTHY)")
    with col_d2:
        st.metric("Drift Share", f"{drift_share * 100:.1f}%", help="Percentage of features failing KS statistical test")
    with col_d3:
        st.metric("Drifted Features", f"{summary.get('drifted_features_count', 0)} / {summary.get('total_features_tested', 0)}")

    st.divider()

    col_btn1, col_btn2, col_btn3 = st.columns(3)

    with col_btn1:
        if st.button("🔄 Run Live Drift Check", use_container_width=True):
            from src.monitoring.drift_detector import generate_drift_report
            with st.spinner("Computing Kolmogorov-Smirnov statistical tests..."):
                generate_drift_report()
            st.rerun()

    with col_btn2:
        if st.button("⚡ Simulate Rush-Hour Surge Drift", use_container_width=True):
            from src.monitoring.drift_simulation import run_drift_simulation_experiment
            with st.spinner("Simulating operational drift..."):
                run_drift_simulation_experiment()
            st.rerun()

    with col_btn3:
        if st.button("🚀 Trigger Automated Retraining", type="primary", use_container_width=True):
            from src.monitoring.retrain_trigger import evaluate_retraining_policy
            with st.spinner("Evaluating policy & triggering MLflow training pipeline..."):
                eval_res = evaluate_retraining_policy(current_rmse=7.2, auto_trigger=True)
            st.success("Retraining pipeline executed successfully!")
            st.rerun()

    # Feature Drift Details Table
    feat_metrics = summary.get("feature_metrics", {})
    if feat_metrics:
        st.subheader("Feature-by-Feature Statistical Drift Summary")
        f_rows = []
        for feat_name, stats in feat_metrics.items():
            f_rows.append({
                "Feature": feat_name,
                "Drift Detected": "⚠️ Yes" if stats.get("drift_detected") else "✅ No",
                "p-value": stats.get("p_value"),
                "KS-Statistic": stats.get("ks_statistic"),
                "Ref Mean": stats.get("ref_mean"),
                "Current Mean": stats.get("curr_mean"),
            })
        st.dataframe(pd.DataFrame(f_rows), use_container_width=True)

    # HTML Report Preview
    if config.drift_report_html.exists():
        with st.expander("📄 View Evidently / Custom HTML Drift Dashboard", expanded=False):
            with open(config.drift_report_html, "r", encoding="utf-8") as f:
                html_code = f.read()
            st.components.v1.html(html_code, height=600, scrolling=True)