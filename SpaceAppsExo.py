# exoplanet_ml_streamlit_enhanced_v2.py
"""
Enhanced Streamlit Exoplanet Detector (Refactor v2)
- Consolidates preprocessing into a Pipeline
- Prevents data leakage by CV on training set only
- Adds robust caching + fallback CSV
- Provides balanced defaults and model export
- Improved similarity metric for KOI comparisons
"""

import streamlit as st
import pandas as pd
import numpy as np
import io
import requests
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
from sklearn.utils import resample
import plotly.graph_objects as go
import os
import tempfile
import warnings

warnings.filterwarnings("ignore")

# -------------------------
# Config / Constants
# -------------------------
st.set_page_config(page_title="Kepler ExoDetect", page_icon="🪐", layout="wide")

TITLE = "Kepler ExoDetect"
SUBTITLE = "A fast, safe RandomForest-based ML demo to flag Kepler KOIs as likely exoplanets or false positives"

API_BASE = "https://exoplanetarchive.ipac.caltech.edu/cgi-bin/nstedAPI/nph-nstedAPI"
API_COLUMNS = [
    'kepid', 'kepoi_name', 'koi_period', 'koi_prad', 'koi_depth',
    'koi_duration', 'koi_steff', 'koi_srad', 'koi_smass', 'koi_disposition'
]

FEATURE_NAMES = ['pl_orbper', 'pl_rade', 'pl_trandep', 'pl_trandur', 'st_teff', 'st_rad', 'st_mass']
DEFAULT_VALUES = {
    'pl_orbper': 10.0,
    'pl_rade': 2.0,
    'pl_trandep': 2000.0,
    'pl_trandur': 3.0,
    'st_teff': 5500.0,
    'st_rad': 1.0,
    'st_mass': 1.0
}

LOCAL_FALLBACK_CSV = "koi_cumulative_backup.csv"  # if you want a local fallback, place a copy here

# Optimized default hyperparameters (reasonable defaults for typical KOI set)
OPT_DEFAULTS = {
    'test_size': 0.2,
    'n_estimators': 200,
    'max_depth': 15,
    'min_samples_split': 4,
    'min_samples_leaf': 2,
    'random_state': 42
}

# -------------------------
# Utility functions
# -------------------------


@st.cache_data(ttl=60*60*6)  # cache for 6 hours
def download_kepler_cumulative():
    """
    Download KOI cumulative table from NASA Exoplanet Archive.
    Falls back to a local CSV if the network/API fails.
    """
    params = {
        'table': 'cumulative',
        'select': ','.join(API_COLUMNS),
        'format': 'csv'
    }

    try:
        r = requests.get(API_BASE, params=params, timeout=30)
        r.raise_for_status()
        df = pd.read_csv(io.StringIO(r.text))
        return df
    except Exception as e:
        # Attempt fallback local CSV if available
        if os.path.exists(LOCAL_FALLBACK_CSV):
            try:
                df = pd.read_csv(LOCAL_FALLBACK_CSV)
                st.warning(f"Could not fetch from NASA API (error: {e}). Using local fallback CSV.")
                return df
            except Exception as e2:
                st.error(f"Failed to read fallback CSV as well: {e2}")
                return None
        else:
            st.error(f"Failed to download KOI table: {e}")
            return None


def map_koi_to_features(koi_df):
    """Map KOI columns to internal feature names and label positives conservatively."""
    if koi_df is None:
        return None

    rename_map = {
        'koi_period': 'pl_orbper',
        'koi_prad': 'pl_rade',
        'koi_depth': 'pl_trandep',
        'koi_duration': 'pl_trandur',
        'koi_steff': 'st_teff',
        'koi_srad': 'st_rad',
        'koi_smass': 'st_mass',
        'koi_disposition': 'koi_disposition'
    }
    df = koi_df.rename(columns=rename_map)

    # Label: CONFIRMED and CANDIDATE => 1, else 0
    df['is_exoplanet'] = np.where(df['koi_disposition'].str.upper().isin(['CONFIRMED', 'CANDIDATE']), 1, 0)

    # Keep only relevant columns
    required = FEATURE_NAMES + ['is_exoplanet']
    for col in FEATURE_NAMES:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    df_clean = df[required].copy().dropna(subset=required)

    if len(df_clean) < 20:
        st.warning("Downloaded KOI table but fewer than 20 usable samples after cleaning.")

    return df_clean


def balance_dataset(df, method="upsample"):
    """
    Balance the dataset. method in {'upsample', 'class_weight', 'none'}.
    - upsample: duplicate minority class to match majority
    - class_weight: leave as is but caller should set class_weight='balanced' in model
    """
    if df is None or len(df) == 0:
        return df

    class_counts = df['is_exoplanet'].value_counts()
    st.write(f"📊 Class distribution before balancing: {dict(class_counts)}")

    if method == "none":
        st.info("Using original class distribution (no balancing).")
        return df
    elif method == "class_weight":
        st.info("Using 'class_weight' option: model will use class weights (no resampling).")
        return df
    elif method == "upsample":
        if len(class_counts) < 2:
            st.info("Only one class present; skipping balancing.")
            return df
        majority_label = class_counts.idxmax()
        minority_label = class_counts.idxmin()
        if class_counts.max() / class_counts.min() <= 1.2:
            st.info("Class distribution fairly balanced already; skipping upsampling.")
            return df

        df_major = df[df['is_exoplanet'] == majority_label]
        df_min = df[df['is_exoplanet'] == minority_label]
        df_min_up = resample(df_min, replace=True, n_samples=len(df_major), random_state=OPT_DEFAULTS['random_state'])
        df_bal = pd.concat([df_major, df_min_up]).sample(frac=1, random_state=OPT_DEFAULTS['random_state']).reset_index(drop=True)
        st.write(f"📊 Class distribution after balancing: {df_bal['is_exoplanet'].value_counts().to_dict()}")
        return df_bal
    else:
        st.error(f"Unknown balancing method: {method}")
        return df


def prepare_features(raw_df, balancing="upsample"):
    """High-level feature preparation and optional balancing. Returns a DataFrame ready for ML."""
    if raw_df is None or raw_df.empty:
        st.error("No KOI data provided.")
        return None

    df = map_koi_to_features(raw_df)
    if df is None or df.empty:
        st.error("No usable rows after mapping/cleaning.")
        return None

    df_bal = balance_dataset(df, method=balancing)
    st.success(f"✅ Prepared dataset with {len(df_bal)} samples.")
    return df_bal


def train_model_pipeline(df, test_size=OPT_DEFAULTS['test_size'], n_estimators=OPT_DEFAULTS['n_estimators'],
                         max_depth=OPT_DEFAULTS['max_depth'], min_samples_split=OPT_DEFAULTS['min_samples_split'],
                         min_samples_leaf=OPT_DEFAULTS['min_samples_leaf'], random_state=OPT_DEFAULTS['random_state'],
                         use_class_weight=False):
    """
    Train a RandomForest inside a Pipeline with StandardScaler.
    Returns: trained pipeline, metrics dict, details dict
    """
    if df is None or len(df) < 10:
        st.error("Not enough data to train (need >=10).")
        return None, None, None

    X = df[FEATURE_NAMES].values
    y = df['is_exoplanet'].astype(int).values

    # Train/test split with stratify
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    # Pipeline: scaler + RF
    rf_kwargs = {
        'n_estimators': n_estimators,
        'max_depth': max_depth,
        'min_samples_split': min_samples_split,
        'min_samples_leaf': min_samples_leaf,
        'random_state': random_state,
        'n_jobs': -1
    }
    if use_class_weight:
        rf_kwargs['class_weight'] = 'balanced'

    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('rf', RandomForestClassifier(**rf_kwargs))
    ])

    pipeline.fit(X_train, y_train)

    # Predictions on test
    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test) if hasattr(pipeline, "predict_proba") else None

    # Metrics
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred, zero_division=0),
        'recall': recall_score(y_test, y_pred, zero_division=0),
        'f1_score': f1_score(y_test, y_pred, zero_division=0),
    }

    # Cross-validation on training set only (no leakage)
    try:
        cv_scores = cross_val_score(pipeline, X_train, y_train, cv=5, scoring='accuracy', n_jobs=-1)
        metrics['cv_mean'] = float(cv_scores.mean())
        metrics['cv_std'] = float(cv_scores.std())
    except Exception as e:
        st.warning(f"Cross-validation failed: {e}")
        metrics['cv_mean'] = None
        metrics['cv_std'] = None
        cv_scores = None

    details = {
        'pipeline': pipeline,
        'X_train': X_train, 'X_test': X_test, 'y_train': y_train, 'y_test': y_test,
        'y_pred': y_pred, 'y_proba': y_proba, 'cv_scores': cv_scores
    }

    return pipeline, metrics, details


def plot_confusion_matrix(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    fig = go.Figure(data=go.Heatmap(
        z=cm,
        x=['Predicted Negative', 'Predicted Positive'],
        y=['Actual Negative', 'Actual Positive'],
        text=cm,
        texttemplate="%{text}",
        colorscale='Blues'
    ))
    fig.update_layout(title="Confusion Matrix", xaxis_title="Predicted", yaxis_title="Actual", height=420)
    return fig


def plot_feature_importance_from_pipeline(pipeline):
    # Extract RF from pipeline and feature importances
    rf = pipeline.named_steps['rf']
    importance = rf.feature_importances_
    idx = np.argsort(importance)
    fig = go.Figure(go.Bar(
        x=importance[idx],
        y=[FEATURE_NAMES[i] for i in idx],
        orientation='h'
    ))
    fig.update_layout(title="Feature Importances (Random Forest)", xaxis_title="Importance", height=420)
    return fig


def find_similar_koi(target_dict, reference_df, top_n=5):
    """
    Uses z-score normalization across reference_df (only positive class rows)
    and Euclidean distance on chosen features to find the top_n most similar KOIs.
    """
    if reference_df is None or reference_df.empty:
        return []

    # Only use positive labelled KOIs for similarity
    ref = reference_df[reference_df['is_exoplanet'] == 1].copy()
    if ref.empty:
        return []

    features = [f for f in FEATURE_NAMES if f in target_dict]
    if not features:
        return []

    # Compute z-score using ref statistics
    ref_stats_mean = ref[features].mean()
    ref_stats_std = ref[features].std().replace(0, 1.0)  # avoid zero division

    # Normalize reference and target
    ref_norm = (ref[features] - ref_stats_mean) / ref_stats_std
    target_vals = np.array([(target_dict[f] - ref_stats_mean[f]) / ref_stats_std[f] for f in features])

    # Compute Euclidean distances
    distances = np.linalg.norm(ref_norm.values - target_vals, axis=1)
    ref = ref.assign(distance=distances)
    best = ref.nsmallest(top_n, 'distance')

    results = []
    for _, row in best.iterrows():
        similarity_score = 1 / (1 + row['distance'])  # higher is more similar, in (0,1]
        results.append({
            'period': row['pl_orbper'],
            'radius': row['pl_rade'],
            'depth': row['pl_trandep'],
            'similarity_score': float(similarity_score)
        })
    return results


# -------------------------
# Streamlit app layout
# -------------------------
def main():
    st.title(f"{TITLE}  🪐")
    st.caption(SUBTITLE)
    st.write("---")

    st.sidebar.title("Navigation")
    mode = st.sidebar.radio("Choose mode", ["Data & Training", "Predict / Analyze", "About & Help"])

    # Session state containers
    if 'pipeline' not in st.session_state:
        st.session_state.pipeline = None
    if 'metrics' not in st.session_state:
        st.session_state.metrics = None
    if 'details' not in st.session_state:
        st.session_state.details = None
    if 'features_df' not in st.session_state:
        st.session_state.features_df = None

    if mode == "Data & Training":
        data_and_training()
    elif mode == "Predict / Analyze":
        predict_and_analyze()
    else:
        about_help()


def data_and_training():
    st.header("1) Data — Download & Inspect")
    if st.button("🔽 Download KOI cumulative table (NASA Exoplanet Archive)"):
        with st.spinner("Downloading..."):
            df_raw = download_kepler_cumulative()
        if df_raw is None:
            st.error("Download failed or no fallback available.")
            return
        st.success(f"Downloaded raw KOI table with {len(df_raw)} rows.")
        st.session_state.raw_koi = df_raw

    if 'raw_koi' in st.session_state:
        if st.checkbox("Show raw KOI table (first 30 rows)"):
            st.dataframe(st.session_state.raw_koi.head(30))

    st.write("---")
    st.header("2) Prepare dataset (cleaning & balancing)")
    bal_method = st.selectbox("Balancing method", ["upsample", "class_weight", "none"], index=0,
                              help="upsample: duplicate minority class; class_weight: rely on RF class weights; none: no balancing")
    if st.button("🧹 Prepare features"):
        if 'raw_koi' not in st.session_state or st.session_state.raw_koi is None:
            st.error("Please download the KOI table first.")
            return
        with st.spinner("Preparing and balancing features..."):
            df_features = prepare_features(st.session_state.raw_koi, balancing=bal_method)
        if df_features is None:
            return
        st.session_state.features_df = df_features
        st.success(f"Prepared dataset with {len(df_features)} rows.")
        if st.checkbox("Preview prepared features (first 40 rows)"):
            st.dataframe(df_features.head(40))

    st.write("---")
    st.header("3) Train model (Pipeline: Scaler → RandomForest)")

    # Hyperparameter controls (optimized defaults pre-filled)
    col1, col2, col3 = st.columns(3)
    with col1:
        test_size = st.slider("Test set fraction", 0.1, 0.4, OPT_DEFAULTS['test_size'], 0.05)
        n_estimators = st.number_input("Number of trees (n_estimators)", min_value=10, max_value=1000,
                                       value=OPT_DEFAULTS['n_estimators'], step=10)
        max_depth = st.selectbox("Max tree depth", [None, 5, 10, 15, 20, 30, 50], index=3)
    with col2:
        min_samples_split = st.number_input("Min samples split", 2, 50, value=OPT_DEFAULTS['min_samples_split'], step=1)
        min_samples_leaf = st.number_input("Min samples leaf", 1, 20, value=OPT_DEFAULTS['min_samples_leaf'], step=1)
        random_state = st.number_input("Random state", 0, 9999, value=OPT_DEFAULTS['random_state'])
    with col3:
        use_class_weight = st.checkbox("Use class_weight='balanced' (instead of upsampling)", value=False)
        show_more = st.checkbox("Show training data distribution before train", value=False)
        if show_more and 'features_df' in st.session_state and st.session_state.features_df is not None:
            st.write(st.session_state.features_df['is_exoplanet'].value_counts().to_dict())

    if st.button("🚀 Train model"):
        if 'features_df' not in st.session_state or st.session_state.features_df is None:
            st.error("Please prepare features first.")
            return

        # If user requested class_weight but original balancing was upsample, that's fine: RF will apply class_weight in pipeline
        with st.spinner("Training pipeline..."):
            pipeline, metrics, details = train_model_pipeline(
                st.session_state.features_df,
                test_size=test_size,
                n_estimators=n_estimators,
                max_depth=max_depth,
                min_samples_split=min_samples_split,
                min_samples_leaf=min_samples_leaf,
                random_state=int(random_state),
                use_class_weight=use_class_weight
            )

        if pipeline is None:
            st.error("Training failed.")
            return

        st.session_state.pipeline = pipeline
        st.session_state.metrics = metrics
        st.session_state.details = details
        st.success("Model trained successfully! ✅")

        # Show metrics
        st.subheader("Key metrics (test set)")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Accuracy", f"{metrics['accuracy']:.2%}")
        c2.metric("Precision", f"{metrics['precision']:.2%}")
        c3.metric("Recall", f"{metrics['recall']:.2%}")
        c4.metric("F1 score", f"{metrics['f1_score']:.2%}")
        if metrics.get('cv_mean') is not None:
            st.write(f"Cross-validation (train only): {metrics['cv_mean']:.2%} ± {metrics['cv_std']:.2%}")

        # Confusion matrix and feature importance
        st.plotly_chart(plot_confusion_matrix(details['y_test'], details['y_pred']), use_container_width=True)
        st.plotly_chart(plot_feature_importance_from_pipeline(pipeline), use_container_width=True)

        st.subheader("Classification Report (test set)")
        cr = classification_report(details['y_test'], details['y_pred'], output_dict=True)
        cr_df = pd.DataFrame(cr).transpose()
        st.dataframe(cr_df.style.format("{:.3f}"))

        # Allow model download
        buf = io.BytesIO()
        pickle.dump({'pipeline': pipeline, 'feature_order': FEATURE_NAMES}, buf)
        buf.seek(0)
        st.download_button("📦 Download trained model (pickle)", data=buf, file_name="exodetect_pipeline.pkl")

    st.write("---")
    st.info("Tip: If you want the model to reflect class rebalancing via upsampling, pick 'upsample' earlier. "
            "If you'd rather not change the dataset and prefer algorithmic balancing, use 'class_weight' and keep 'none' in preparation.")

def predict_and_analyze():
    st.header("Predict / Analyze a Candidate")

    if st.session_state.get('pipeline') is None:
        st.warning("No trained model found. Please train a model first (Data & Training tab).")
        return

    pipeline = st.session_state.pipeline
    confirmed_snapshot = st.session_state.get('features_df')  # used for similarity search

    st.markdown("Select available parameters for your target object. Leave unknown parameters unchecked.")
    col1, col2 = st.columns(2)

    with col1:
        use_p = st.checkbox("Orbital Period (days)", value=True)
        if use_p:
            pl_orbper = st.number_input("Period (days)", min_value=0.001, value=float(DEFAULT_VALUES['pl_orbper']), step=0.1)
        else:
            pl_orbper = None

        use_r = st.checkbox("Planet Radius (Earth radii)", value=True)
        if use_r:
            pl_rade = st.number_input("Radius (Earth radii)", min_value=0.01, value=float(DEFAULT_VALUES['pl_rade']), step=0.1)
        else:
            pl_rade = None

        use_depth = st.checkbox("Transit Depth (ppm)", value=True)
        if use_depth:
            pl_trandep = st.number_input("Transit depth (ppm)", min_value=1.0, value=float(DEFAULT_VALUES['pl_trandep']), step=10.0)
        else:
            pl_trandep = None

        use_dur = st.checkbox("Transit Duration (hours)", value=True)
        if use_dur:
            pl_trandur = st.number_input("Transit duration (hours)", min_value=0.01, value=float(DEFAULT_VALUES['pl_trandur']), step=0.1)
        else:
            pl_trandur = None

    with col2:
        use_teff = st.checkbox("Stellar Teff (K)", value=True)
        if use_teff:
            st_teff = st.number_input("Stellar Teff (K)", min_value=1000.0, max_value=50000.0, value=float(DEFAULT_VALUES['st_teff']), step=50.0)
        else:
            st_teff = None

        use_srad = st.checkbox("Stellar Radius (R☉)", value=True)
        if use_srad:
            st_rad = st.number_input("Stellar Radius (R☉)", min_value=0.01, value=float(DEFAULT_VALUES['st_rad']), step=0.01)
        else:
            st_rad = None

        use_smass = st.checkbox("Stellar Mass (M☉)", value=True)
        if use_smass:
            st_mass = st.number_input("Stellar Mass (M☉)", min_value=0.01, value=float(DEFAULT_VALUES['st_mass']), step=0.01)
        else:
            st_mass = None

    # Build input vector and defaults
    used_features = []
    feature_values = {}
    for f, val in [('pl_orbper', pl_orbper), ('pl_rade', pl_rade), ('pl_trandep', pl_trandep),
                   ('pl_trandur', pl_trandur), ('st_teff', st_teff), ('st_rad', st_rad), ('st_mass', st_mass)]:
        if val is not None:
            used_features.append(f)
            feature_values[f] = float(val)

    st.write(f"Using features: {', '.join(used_features) if used_features else 'None (will use all defaults)'}")

    if st.button("🔍 Analyze candidate"):
        if not used_features:
            st.error("Please provide at least one parameter.")
            return

        # Prepare full input using defaults for missing features
        input_vec = [feature_values.get(f, DEFAULT_VALUES[f]) for f in FEATURE_NAMES]
        input_arr = np.array([input_vec])

        # Predict
        proba = pipeline.predict_proba(input_arr)[0]
        pred = pipeline.predict(input_arr)[0]

        st.subheader("Prediction")
        if pred == 1:
            st.success("PREDICTION: Likely Exoplanet ✅")
        else:
            st.error("PREDICTION: Likely False Positive ❌")
        st.metric("Confidence (top class)", f"{100.0 * np.max(proba):.1f}%")
        st.write(f"Probability (Not Exoplanet): {proba[0]*100:.2f}%")
        st.write(f"Probability (Exoplanet): {proba[1]*100:.2f}%")

        # Feature importance snapshot
        st.subheader("Model feature importances (global)")
        fi = pipeline.named_steps['rf'].feature_importances_
        fi_df = pd.DataFrame({'feature': FEATURE_NAMES, 'importance': fi}).sort_values('importance', ascending=False)
        st.table(fi_df.style.format({'importance': "{:.4f}"}))

        # Similar KOIs
        st.subheader("Top similar confirmed/candidate KOIs (from training snapshot)")
        sim = find_similar_koi(feature_values, confirmed_snapshot, top_n=5)
        if sim:
            for i, s in enumerate(sim, 1):
                st.write(f"{i}. P={s['period']:.2f} d, R={s['radius']:.2f} R⊕ — similarity {s['similarity_score']:.3f}")
        else:
            st.info("No similar KOIs found (check that you trained the model on a non-empty dataset).")

        # Optional: SHAP explanation if package available
        try:
            import shap
            st.subheader("Local explanation (SHAP) — requires shap package")
            explainer = shap.Explainer(pipeline.named_steps['rf'], pipeline.named_steps['scaler'].transform(confirmed_snapshot[FEATURE_NAMES].values))
            shap_values = explainer(pipeline.named_steps['scaler'].transform(input_arr))
            shap_html = shap.plots.waterfall(shap_values[0], show=False)
            st.write("SHAP output shown in a new tab may be more useful; local inline rendering depends on your environment.")
        except Exception:
            # If shap not available, skip silently (non-blocking)
            pass


def about_help():
    st.header("About Kepler ExoDetect — friendly guide")
    st.markdown("""
**What this app is**  
Kepler ExoDetect is a compact demo and research-friendly tool that pulls KOI (Kepler Objects of Interest) data and trains a RandomForest model to separate likely exoplanets (CONFIRMED + CANDIDATE) from false positives.

**Why this approach?**  
- We use physically meaningful inputs (orbital period, radius, transit depth/duration, and host star properties).  
- A RandomForest is robust on tabular data and easy to interpret via feature importances.  
- We build a `Pipeline` with a `StandardScaler` so feature scaling is always applied consistently during training and prediction.

**How to use (quick steps)**  
1. **Data & Training → Download** the KOI table (or use the included fallback CSV).  
2. **Prepare features** (you can upsample the minority class or use class weights).  
3. **Train** the model — optimized defaults are pre-filled so you usually only press Train.  
4. **Predict / Analyze** any candidate by entering the parameters you have. The model will show prediction + probability and list similar KOIs found in the training snapshot.

**Design choices & honest caveats**  
- **No magical accuracy**: This is a demo classifier. Real exoplanet vetting uses time-series modeling, astrophysical vetting metrics, human follow-up, and domain-specific heuristics.  
- **Labels**: We label both *CONFIRMED* and *CANDIDATE* as positives — this helps the model learn potential planets but can inflate apparent performance. Treat the model as a triage tool, not a definitive classifier.  
- **Data quality matters**: Missing features are filled with sensible defaults when predicting; better inputs → better predictions.  
- **Explainability**: We provide global feature importances and optional SHAP-based local explanations (if `shap` installed).

**Quick tips**  
- If your precision is low but recall is high, the model flags many candidates (useful for triage). If you need fewer false positives, increase `min_samples_leaf` or lower `max_depth`.  
- To replicate results exactly, set `Random state` and download the pickle after training.
""")

if __name__ == "__main__":
    main()