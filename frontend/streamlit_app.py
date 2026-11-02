"""Tea Flush Planner - my Streamlit front end for the tea harvest-readiness model.

Run from the repository root:
    streamlit run frontend/streamlit_app.py

What the app does
  * Predicts whether a Sri Lankan tea region is harvest-ready for a month's weather,
    and explains the decision with the Explainable Boosting Machine's exact contributions.
  * What-if simulator, a 12-month plucking calendar with climate scenarios, batch scoring
    of a CSV, a data explorer, the model comparison and the global explanations.
  * Works fully offline with the saved model. If API_URL is set (my Azure deployment), it can
    send predictions to the live FastAPI service instead, and falls back to the local model
    if the API cannot be reached.
"""

from __future__ import annotations

import io
import json
import os
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# ---------------------------------------------------------------------------
# Paths: the app lives in <repo>/frontend, the shared feature code in <repo>/src.
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.features import (  # noqa: E402
    FEATURE_LABELS,
    MODEL_FEATURES,
    MONTH_NAMES,
    NDVI_LOOKUP,
    REGIONS,
    TRAINING_RANGES,
    build_features,
    out_of_training_range,
    season_of,
    validate_inputs,
)

DATA_PATH = ROOT / "data" / "raw" / "sri_lanka_tea_tri_gov_ds.csv"
MODEL_CANDIDATES = [
    Path(os.getenv("MODEL_PATH", ROOT / "models" / "tea_harvest_ebm.joblib")),
    ROOT / "models" / "model.joblib",
]
MET_DIR = ROOT / "reports" / "metrics"
FIG_DIR = ROOT / "reports" / "figures"
API_URL = os.getenv("API_URL", "").rstrip("/")
THRESHOLD = 0.5

# ---------------------------------------------------------------------------
# Design tokens (kept identical to the notebook's plot palette)
# ---------------------------------------------------------------------------
ESTATE = "#12372A"  # deep estate green: sidebar, headings
LEAF = "#3E7C3A"  # plucking-leaf green: "ready"
LIQUOR = "#C98E1D"  # tea-liquor gold: "not ready"
MIST = "#EEF3EA"  # misty-leaf background
INK = "#1E2B24"  # body text
MUTED = "#5B6B61"  # secondary text
LINE = "#D5DED3"  # hairlines and borders
NOT_READY = "#93650F"  # darker tea-liquor gold: white text stays readable on it

st.set_page_config(
    page_title="Tea Flush Planner", page_icon="🍃", layout="wide", initial_sidebar_state="expanded"
)

CSS = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Bricolage+Grotesque:opsz,wght@12..96,500;12..96,700&family=Source+Sans+3:wght@400;600&display=swap');
html, body, [class*="css"], .stMarkdown, p, li, label, .stTextInput, .stNumberInput {{
  font-family: 'Source Sans 3', system-ui, sans-serif; color: {INK};
}}
h1, h2, h3, h4 {{ font-family: 'Bricolage Grotesque', 'Source Sans 3', sans-serif !important;
  color: {ESTATE} !important; letter-spacing: -0.01em; }}
h1 {{ font-size: 2.3rem !important; font-weight: 700 !important; line-height: 1.1 !important; }}
h2 {{ font-size: 1.55rem !important; font-weight: 700 !important; }}
h3 {{ font-size: 1.15rem !important; font-weight: 500 !important; }}
.block-container {{ padding-top: 2.2rem; max-width: 1200px; }}
[data-testid="stSidebar"] {{ background: {ESTATE}; }}
[data-testid="stSidebar"] * {{ color: #E8F0E6 !important; }}
[data-testid="stSidebar"] a[aria-current="page"] {{ background: rgba(255,255,255,0.12); border-radius: 6px; }}
[data-testid="stSidebar"] hr {{ border-color: rgba(255,255,255,0.18); }}
.lede {{ font-size: 1.12rem; color: {MUTED}; max-width: 68ch; line-height: 1.55; }}
.panel {{ background: #FFFFFF; border: 1px solid {LINE}; border-radius: 10px; padding: 1.1rem 1.25rem; }}
.verdict {{ border-radius: 12px; padding: 1.2rem 1.4rem; color: #FFFFFF; }}
.verdict h3 {{ color: #FFFFFF !important; margin: 0 0 .25rem 0; font-size: 1.5rem !important; font-weight: 700 !important; }}
.verdict p {{ color: #FFFFFF; margin: 0; font-size: 1.02rem; }}
.ready {{ background: {LEAF}; }}
.notready {{ background: #8A5F0F; }}
.note {{ border-left: 4px solid {LIQUOR}; background: #FBF5E8; padding: .7rem 1rem; border-radius: 0 8px 8px 0; }}
.ribbon {{ display: grid; grid-template-columns: repeat(12, 1fr); gap: 6px; margin: .4rem 0 1.1rem 0; }}
.cell {{ border-radius: 8px; padding: .55rem .2rem; text-align: center; font-weight: 600; font-size: .9rem; }}
.cell small {{ display: block; font-weight: 400; font-size: .78rem; opacity: .9; }}
.legend-dot {{ display:inline-block; width:.8rem; height:.8rem; border-radius:3px; vertical-align:-1px; margin-right:.3rem; }}
div[data-testid="stMetricValue"] {{ font-family: 'Bricolage Grotesque', sans-serif; color: {ESTATE}; }}
.stButton>button, .stDownloadButton>button, .stFormSubmitButton>button {{
  border-radius: 8px; font-weight: 600; border: 1px solid {LEAF}; }}
.stFormSubmitButton>button, .stFormSubmitButton>button p {{ background: {LEAF}; color: #FFFFFF !important; }}
.stFormSubmitButton>button:hover, .stFormSubmitButton>button:hover p {{ background: {ESTATE}; color: #FFFFFF !important; border-color: {ESTATE}; }}
.pill {{ display:inline-block; padding:.15rem .7rem; border-radius:999px; font-weight:600; font-size:.95rem; }}
:focus-visible {{ outline: 3px solid {LIQUOR} !important; outline-offset: 2px; }}
@media (max-width: 700px) {{ .ribbon {{ grid-template-columns: repeat(6, 1fr); }} h1 {{ font-size: 1.8rem !important; }} }}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Cached loaders with clear error messages
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading the model…")
def load_model():
    """Load the first model file that exists. Returns (model, path) or (None, error message)."""
    for path in MODEL_CANDIDATES:
        if path.exists():
            try:
                return joblib.load(path), str(path.relative_to(ROOT))
            except Exception as err:  # version mismatch, corrupted file, ...
                return None, f"{path.name} could not be loaded ({type(err).__name__}: {err})"
    return (
        None,
        "No model file found. Run the notebook (or `make repro`) to create models/tea_harvest_ebm.joblib.",
    )


@st.cache_data(show_spinner=False)
def load_data() -> pd.DataFrame | None:
    try:
        df = pd.read_csv(DATA_PATH)
        df["month_name"] = df["month"].map(lambda m: MONTH_NAMES[m - 1])
        return df
    except FileNotFoundError:
        return None


@st.cache_data(show_spinner=False)
def load_json(name: str) -> dict | None:
    try:
        return json.loads((MET_DIR / name).read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return None


@st.cache_data(show_spinner=False)
def load_model_card() -> dict | None:
    try:
        return json.loads((ROOT / "models" / "model_card.json").read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return None


@st.cache_data(show_spinner=False)
def load_csv(name: str) -> pd.DataFrame | None:
    try:
        return pd.read_csv(MET_DIR / name, index_col=0)
    except FileNotFoundError:
        return None


@st.cache_data(show_spinner=False)
def climatology(df: pd.DataFrame) -> pd.DataFrame:
    """Median weather for each region and month: the 'typical year' used for defaults."""
    cols = ["rainfall_mm", "temperature_c", "humidity_percent", "soil_moisture_percent"]
    return df.groupby(["region", "month"])[cols].median().round(1)


model, model_source = load_model()
df = load_data()
clim = climatology(df) if df is not None else None


# ---------------------------------------------------------------------------
# Prediction helpers
# ---------------------------------------------------------------------------
def predict_local(rows: pd.DataFrame) -> np.ndarray:
    return model.predict_proba(build_features(rows))[:, 1]


def explain_local(row: pd.DataFrame) -> tuple[float, pd.DataFrame]:
    """EBM's exact local explanation: intercept + contribution of every term (log-odds)."""
    ebm = model.named_steps["model"]
    X_imp = model.named_steps["impute"].transform(build_features(row))
    data = ebm.explain_local(X_imp).data(0)
    contrib = pd.DataFrame({"term": data["names"], "contribution": data["scores"]})
    contrib["label"] = contrib["term"].map(pretty_term)
    return float(ebm.intercept_[0]), contrib.sort_values(
        "contribution", key=np.abs, ascending=False
    )


def predict_via_api(payload: dict) -> dict | None:
    """Call the live FastAPI /explain endpoint. Returns None (and warns) if it is unreachable."""
    try:
        import requests

        resp = requests.post(f"{API_URL}/explain", json=payload, timeout=20)
        if resp.status_code == 422:
            st.error(
                "The live API rejected these values (data contract check). "
                f"Details: {resp.json().get('detail')}"
            )
            return None
        resp.raise_for_status()
        return resp.json()
    except Exception as err:
        st.warning(f"Live API not reachable ({type(err).__name__}). Using the local model instead.")
        return None


def pretty_term(term: str) -> str:
    return " × ".join(FEATURE_LABELS.get(t.strip(), t) for t in term.split("&"))


def advice(prob: float, row: dict, contrib: pd.DataFrame | None) -> list[str]:
    """Plain-language guidance for an estate manager, based on the strongest drivers."""
    tips = []
    if prob >= 0.8:
        tips.append("Schedule full plucking rounds; the flush is expected this month.")
    elif prob >= THRESHOLD:
        tips.append(
            "Plan plucking, but walk the fields first: the model is only moderately confident."
        )
    elif prob >= 0.3:
        tips.append(
            "Borderline month. Keep a reduced gang on standby and re-check after the next rains."
        )
    else:
        tips.append(
            "Hold the plucking gangs or move them to other fields; growth is unlikely to be ready."
        )
    if row["rainfall_mm"] < 80:
        tips.append(
            "Rainfall is below about 80 mm, the level the model treats as the main dry-month gate."
        )
    if row["humidity_percent"] > 90:
        tips.append(
            "Humidity above about 90% (misty, cloudy weather) holds back the flush and raises blister-blight risk."
        )
    if row["temperature_c"] < 16:
        tips.append("Cool temperatures slow shoot growth; expect a slow up-country month.")
    if contrib is not None and len(contrib):
        top = contrib.iloc[0]
        direction = "towards ready" if top.contribution > 0 else "towards not ready"
        tips.append(f"The strongest single factor is {top.label.lower()}, pushing {direction}.")
    return tips


def plotly_layout(fig: go.Figure, height: int = 360, **kw) -> go.Figure:
    fig.update_layout(
        height=height,
        margin=dict(l=10, r=10, t=40, b=10),
        paper_bgcolor="#FFFFFF",
        plot_bgcolor="#FFFFFF",
        font=dict(family="Source Sans 3, sans-serif", color=INK, size=13),
        title_font=dict(family="Bricolage Grotesque, sans-serif", color=ESTATE, size=16),
        **kw,
    )
    fig.update_xaxes(gridcolor="#EEF1EC", zerolinecolor=LINE)
    fig.update_yaxes(gridcolor="#EEF1EC", zerolinecolor=LINE)
    return fig


def contribution_chart(intercept: float, contrib: pd.DataFrame, top_n: int = 8) -> go.Figure:
    c = contrib.head(top_n).iloc[::-1]
    fig = go.Figure(
        go.Bar(
            x=c["contribution"],
            y=c["label"],
            orientation="h",
            marker_color=[LEAF if v > 0 else LIQUOR for v in c["contribution"]],
            text=[f"{v:+.2f}" for v in c["contribution"]],
            textposition="outside",
            cliponaxis=False,
            hovertemplate="%{y}: %{x:+.3f} log-odds<extra></extra>",
        )
    )
    fig.add_vline(x=0, line_color=MUTED, line_width=1)
    lo, hi = float(min(0, c["contribution"].min())), float(max(0, c["contribution"].max()))
    pad = 0.22 * max(hi - lo, 0.5)  # room for the value labels
    fig.update_xaxes(range=[lo - pad, hi + pad])
    return plotly_layout(
        fig,
        height=90 + 38 * len(c),
        title=f"Why: contribution of each factor (base log-odds {intercept:+.2f})",
        xaxis_title="log-odds (+ towards ready, − towards not ready)",
    )


def gauge(prob: float) -> go.Figure:
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=prob * 100,
            number={"suffix": "%", "font": {"size": 40, "color": ESTATE}},
            gauge={
                "axis": {"range": [0, 100], "tickwidth": 1},
                "bar": {"color": LEAF if prob >= THRESHOLD else LIQUOR, "thickness": 0.3},
                "steps": [
                    {"range": [0, 50], "color": "#F6EBD3"},
                    {"range": [50, 100], "color": "#DCEBD8"},
                ],
                "threshold": {"line": {"color": ESTATE, "width": 3}, "value": THRESHOLD * 100},
            },
            title={"text": "Chance the region is harvest-ready", "font": {"size": 15}},
        )
    )
    fig = plotly_layout(fig, height=290)
    fig.update_layout(margin=dict(l=30, r=30, t=60, b=10))
    return fig


def ribbon_html(probs: list[float], subtitle: list[str] | None = None) -> str:
    """The 12-month plucking ribbon: one cell per month, coloured by readiness."""
    cells = []
    for i, p in enumerate(probs):
        ready = p >= THRESHOLD
        # Blend from gold (not ready) to green (ready); text stays high-contrast.
        bg = LEAF if p >= 0.8 else ("#A9C79A" if ready else ("#EBCB86" if p >= 0.3 else NOT_READY))
        fg = "#FFFFFF" if p >= 0.8 or p < 0.3 else INK
        sub = subtitle[i] if subtitle else f"{p:.0%}"
        cells.append(
            f'<div class="cell" style="background:{bg};color:{fg}" title="{MONTH_NAMES[i]}: {p:.0%}">'
            f"{MONTH_NAMES[i]}<small>{sub}</small></div>"
        )
    return f'<div class="ribbon">{"".join(cells)}</div>'


def legend_html() -> str:
    items = [
        (LEAF, "Ready, high confidence"),
        ("#A9C79A", "Ready"),
        ("#EBCB86", "Borderline"),
        (NOT_READY, "Not ready"),
    ]
    return " &nbsp; ".join(
        f'<span class="legend-dot" style="background:{c}"></span>{t}' for c, t in items
    )


def require_model() -> bool:
    if model is None:
        st.error(f"The model is not available: {model_source}")
        return False
    return True


def require_data() -> bool:
    if df is None:
        st.error(
            f"The dataset was not found at {DATA_PATH.relative_to(ROOT)}. Run `dvc pull` or copy the CSV there."
        )
        return False
    return True


def region_caption(region: str, month: int) -> str:
    info = REGIONS[region]
    season = season_of(month)
    ndvi = NDVI_LOOKUP[(info["zone"], season)]
    return (
        f"{region} sits at {info['elevation_m']:,} m ({info['zone']}). "
        f"{MONTH_NAMES[month - 1]} is in the {season} season "
        f"({'NE monsoon' if season == 'Maha' else 'SW monsoon'}); typical NDVI {ndvi}."
    )


# ===========================================================================
# Pages
# ===========================================================================
def page_home():
    st.title("When will the next flush be ready?")
    st.markdown(
        '<p class="lede">Tea Flush Planner tells Sri Lankan estate managers whether a tea region is '
        "ready for plucking in a given month, and which weather conditions are driving that call. "
        "It is built on 24 years of monthly records from 11 tea-growing regions.</p>",
        unsafe_allow_html=True,
    )
    if not (require_data() and require_model()):
        return

    region = st.selectbox(
        "Region",
        list(REGIONS),
        index=0,
        key="home_region",
        help="Regions are listed from the highest to the lowest elevation.",
    )
    hist = df[df.region == region].groupby("month")["harvest_ready"].mean().tolist()
    typical = (
        clim.loc[region]
        .reset_index()
        .assign(region=region, elevation_m=REGIONS[region]["elevation_m"])
    )
    model_probs = predict_local(typical).tolist()

    st.markdown(f"#### Plucking ribbon for {region}: share of months that were ready, 2001–2024")
    st.markdown(ribbon_html(hist), unsafe_allow_html=True)
    st.markdown(f"#### Model's call for a typical year in {region}")
    st.markdown(ribbon_html(model_probs), unsafe_allow_html=True)
    st.caption(legend_html(), unsafe_allow_html=True)

    st.divider()
    c1, c2, c3, c4 = st.columns(4)
    card = load_model_card() or {}
    tm = card.get("test_metrics", {})
    c1.metric("Region-months analysed", f"{len(df):,}")
    c2.metric("Regions", "11", help="High-grown, mid-grown and low-grown districts")
    c3.metric("Test accuracy", f"{tm.get('Accuracy', np.nan):.1%}" if tm else "—")
    c4.metric("Test ROC-AUC", f"{tm.get('ROC-AUC', np.nan):.4f}" if tm else "—")

    left, right = st.columns([1.15, 1])
    with left:
        st.subheader("Where the regions are")
        pts = pd.DataFrame([{"region": r, **v} for r, v in REGIONS.items()])
        rate = df.groupby("region")["harvest_ready"].mean()
        pts["ready_rate"] = pts["region"].map(rate)
        fig = go.Figure(
            go.Scattermap(
                lat=pts.lat,
                lon=pts.lon,
                mode="markers+text",
                text=pts.region,
                textposition="top right",
                marker=dict(
                    size=12 + pts.elevation_m / 110,
                    color=pts.ready_rate,
                    cmin=0.35,
                    cmax=0.9,
                    colorscale=[[0, LIQUOR], [0.5, "#B9C77A"], [1, LEAF]],
                    showscale=True,
                    colorbar=dict(title="Ready<br>rate", tickformat=".0%"),
                ),
                customdata=np.stack([pts.elevation_m, pts.zone, pts.ready_rate], axis=1),
                hovertemplate="<b>%{text}</b><br>%{customdata[0]} m, %{customdata[1]}"
                "<br>Ready in %{customdata[2]:.0%} of months<extra></extra>",
            )
        )
        fig.update_layout(
            map=dict(style="open-street-map", center=dict(lat=6.85, lon=80.7), zoom=6.6)
        )
        st.plotly_chart(plotly_layout(fig, height=430), width="stretch")
        st.caption(
            "Marker size grows with elevation; colour shows how often the region is harvest-ready."
        )
    with right:
        st.subheader("How to use it")
        st.markdown(
            "- **Predict a month** with this month's rainfall, temperature, humidity and soil moisture.\n"
            "- **Try what-ifs** to see how much rain would make a dry month ready.\n"
            "- **Plan the year** with the 12-month calendar, including drought and warming scenarios.\n"
            "- **Score many months** at once by uploading a CSV.\n"
            "- **Check the model**: how it was compared, validated and what it has learned."
        )
        st.markdown(
            '<div class="note">The model is decision support. It reproduces the readiness rules in the '
            "TRI-style dataset with about 99% accuracy, but a field inspection should still confirm "
            "the flush before gangs are sent out.</div>",
            unsafe_allow_html=True,
        )


def page_predict():
    st.title("Predict a month")
    st.markdown(
        '<p class="lede">Enter the month\'s conditions for a region. Fields start at the typical value '
        "for that region and month, so you only need to change what you measured.</p>",
        unsafe_allow_html=True,
    )
    if not (require_model() and require_data()):
        return

    c1, c2, c3 = st.columns([1.3, 1, 1])
    region = c1.selectbox("Region", list(REGIONS), key="p_region")
    month = c2.selectbox(
        "Month",
        list(range(1, 13)),
        format_func=lambda m: MONTH_NAMES[m - 1],
        index=3,
        key="p_month",
    )
    year = c3.number_input(
        "Year",
        min_value=1990,
        max_value=2100,
        value=2025,
        step=1,
        help="Used only for the record; the model does not use the year.",
    )
    st.caption(region_caption(region, month))
    typical = clim.loc[(region, month)]

    use_api = False
    if API_URL:
        use_api = st.toggle(
            "Send to the live Azure API",
            value=True,
            help=f"Calls {API_URL}/explain. Falls back to the local model if unreachable.",
        )

    with st.form("predict_form"):
        a, b = st.columns(2)
        rainfall = a.number_input(
            "Monthly rainfall (mm)", 0.0, 1200.0, float(typical.rainfall_mm), 5.0
        )
        temperature = b.number_input(
            "Mean temperature (°C)",
            5.0,
            40.0,
            float(typical.temperature_c),
            0.1,
            help="Elevation-adjusted mean for the estate, as recorded by TRI stations.",
        )
        humidity = a.number_input(
            "Relative humidity (%)", 20.0, 100.0, float(typical.humidity_percent), 0.5
        )
        soil = b.number_input(
            "Soil moisture (% of field capacity)",
            0.0,
            100.0,
            float(typical.soil_moisture_percent),
            1.0,
        )
        submitted = st.form_submit_button("Predict readiness", width="stretch")

    if not submitted:
        st.info("Adjust the values and select **Predict readiness**.")
        return

    row = {
        "region": region,
        "year": int(year),
        "month": int(month),
        "rainfall_mm": rainfall,
        "temperature_c": temperature,
        "humidity_percent": humidity,
        "soil_moisture_percent": soil,
        "elevation_m": REGIONS[region]["elevation_m"],
    }
    problems = validate_inputs(row)
    if problems:
        for p in problems:
            st.error(p)
        return
    flagged = out_of_training_range(row)
    if flagged:
        st.warning(
            "Outside the range seen in training, so treat the result with care: "
            + ", ".join(flagged)
        )

    source = "local model"
    result = predict_via_api({k: row[k] for k in row if k != "elevation_m"}) if use_api else None
    try:
        if result:
            prob = result["probability"]
            intercept = result["intercept"]
            contrib = pd.DataFrame(result["contributions"])
            contrib["label"] = contrib["term"].map(pretty_term)
            source = f"live API ({result.get('model_version', '?')})"
        else:
            row_df = pd.DataFrame([row])
            prob = float(predict_local(row_df)[0])
            intercept, contrib = explain_local(row_df)
    except Exception as err:
        st.error(f"Prediction failed: {type(err).__name__}: {err}")
        return

    ready = prob >= THRESHOLD
    left, right = st.columns([1, 1.25])
    with left:
        st.markdown(
            f'<div class="verdict {"ready" if ready else "notready"}"><h3>'
            f"{'Ready for plucking' if ready else 'Not ready yet'}</h3>"
            f"<p>{region}, {MONTH_NAMES[month - 1]} {year}: {prob:.1%} chance of a ready flush</p></div>",
            unsafe_allow_html=True,
        )
        st.plotly_chart(gauge(prob), width="stretch")
        st.caption(f"Prediction from the {source}. Decision threshold {THRESHOLD:.0%}.")
    with right:
        st.plotly_chart(contribution_chart(intercept, contrib), width="stretch")

    st.subheader("What this means for the estate")
    for tip in advice(prob, row, contrib):
        st.markdown(f"- {tip}")

    out = pd.DataFrame(
        [
            {
                **row,
                "probability_ready": round(prob, 4),
                "prediction": "ready" if ready else "not ready",
            }
        ]
    )
    st.download_button(
        "Download this prediction (CSV)",
        out.to_csv(index=False).encode(),
        file_name=f"tea_prediction_{region.replace(' ', '_')}_{year}_{month:02d}.csv",
        mime="text/csv",
    )
    hist = st.session_state.setdefault("history", [])
    hist.append(out.iloc[0].to_dict())
    with st.expander(f"Predictions this session ({len(hist)})"):
        st.dataframe(pd.DataFrame(hist), width="stretch", hide_index=True)


def page_whatif():
    st.title("What-if simulator")
    st.markdown(
        '<p class="lede">Move one condition and watch the chance of a ready month change. '
        "The curves show where the model's thresholds sit for this region and month.</p>",
        unsafe_allow_html=True,
    )
    if not (require_model() and require_data()):
        return
    c1, c2 = st.columns(2)
    region = c1.selectbox("Region", list(REGIONS), key="w_region")
    month = c2.select_slider(
        "Month",
        options=list(range(1, 13)),
        value=1,
        format_func=lambda m: MONTH_NAMES[m - 1],
        key="w_month",
    )
    t = clim.loc[(region, month)]
    s1, s2, s3, s4 = st.columns(4)
    rain = s1.slider("Rainfall (mm)", 0.0, 750.0, float(t.rainfall_mm), 5.0)
    temp = s2.slider("Temperature (°C)", 10.0, 32.0, float(t.temperature_c), 0.1)
    hum = s3.slider("Humidity (%)", 60.0, 95.0, float(t.humidity_percent), 0.5)
    soil = s4.slider("Soil moisture (%)", 45.0, 100.0, float(t.soil_moisture_percent), 1.0)
    base = {
        "region": region,
        "month": month,
        "rainfall_mm": rain,
        "temperature_c": temp,
        "humidity_percent": hum,
        "soil_moisture_percent": soil,
        "elevation_m": REGIONS[region]["elevation_m"],
    }
    prob = float(predict_local(pd.DataFrame([base]))[0])
    m1, m2 = st.columns([1, 3])
    m1.metric("Chance of a ready month", f"{prob:.0%}")
    pill_bg, pill_text = (LEAF, "Ready") if prob >= THRESHOLD else (NOT_READY, "Not ready")
    m2.markdown(
        f'<div style="padding-top:2.1rem"><span class="pill" style="background:{pill_bg};color:#fff">'
        f"{pill_text}</span></div>",
        unsafe_allow_html=True,
    )

    sweeps = {
        "rainfall_mm": np.linspace(0, 750, 120),
        "temperature_c": np.linspace(10, 32, 90),
        "humidity_percent": np.linspace(60, 95, 90),
        "soil_moisture_percent": np.linspace(45, 100, 80),
    }
    cols = st.columns(2)
    for i, (feat, grid) in enumerate(sweeps.items()):
        rows = pd.DataFrame([base] * len(grid))
        rows[feat] = grid
        p = predict_local(rows)
        fig = go.Figure()
        fig.add_hrect(
            y0=THRESHOLD, y1=1, fillcolor="#DCEBD8", opacity=0.5, line_width=0, layer="below"
        )
        fig.add_trace(
            go.Scatter(
                x=grid,
                y=p,
                mode="lines",
                line=dict(color=ESTATE, width=3),
                hovertemplate="%{x:.1f} → %{y:.0%}<extra></extra>",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=[base[feat]],
                y=[prob],
                mode="markers",
                marker=dict(size=13, color=LIQUOR, line=dict(color="#fff", width=2)),
                hovertemplate="current: %{x:.1f}<extra></extra>",
            )
        )
        fig.update_yaxes(range=[0, 1.02], tickformat=".0%")
        cols[i % 2].plotly_chart(
            plotly_layout(
                fig,
                height=280,
                showlegend=False,
                title=FEATURE_LABELS[feat],
                yaxis_title="P(ready)",
            ),
            width="stretch",
        )
    st.caption(
        "Green band = the model calls the month ready. The gold dot is your current setting."
    )


def page_calendar():
    st.title("Plucking calendar")
    st.markdown(
        '<p class="lede">A 12-month outlook for one region built from its typical weather, with optional '
        "drought or warming scenarios. Use it to plan labour and factory capacity for the year.</p>",
        unsafe_allow_html=True,
    )
    if not (require_model() and require_data()):
        return
    c1, c2, c3 = st.columns([1.2, 1, 1])
    region = c1.selectbox("Region", list(REGIONS), key="c_region")
    rain_change = c2.slider(
        "Rainfall change (%)",
        -60,
        60,
        0,
        5,
        help="Apply to every month, e.g. −30% for a drought year",
    )
    temp_change = c3.slider(
        "Temperature change (°C)", -2.0, 3.0, 0.0, 0.25, help="e.g. +1.5 °C warming"
    )
    base = (
        clim.loc[region]
        .reset_index()
        .assign(region=region, elevation_m=REGIONS[region]["elevation_m"])
    )
    scen = base.copy()
    scen["rainfall_mm"] = (scen["rainfall_mm"] * (1 + rain_change / 100)).clip(lower=0)
    scen["temperature_c"] = scen["temperature_c"] + temp_change
    p_base, p_scen = predict_local(base), predict_local(scen)

    st.markdown("#### Typical year")
    st.markdown(ribbon_html(list(p_base)), unsafe_allow_html=True)
    if rain_change or temp_change:
        st.markdown(f"#### Scenario: rainfall {rain_change:+d}%, temperature {temp_change:+.2f} °C")
        st.markdown(ribbon_html(list(p_scen)), unsafe_allow_html=True)
    st.caption(legend_html(), unsafe_allow_html=True)

    a, b, c = st.columns(3)
    a.metric("Ready months, typical year", int((p_base >= THRESHOLD).sum()))
    b.metric(
        "Ready months, scenario",
        int((p_scen >= THRESHOLD).sum()),
        delta=int((p_scen >= THRESHOLD).sum() - (p_base >= THRESHOLD).sum()),
    )
    c.metric(
        "Expected plucking months", f"{p_scen.sum():.1f}", help="Sum of the monthly probabilities"
    )

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=MONTH_NAMES,
            y=base["rainfall_mm"],
            name="Typical rainfall",
            marker_color="#B7CDB2",
            opacity=0.8,
        )
    )
    fig.add_trace(
        go.Scatter(
            x=MONTH_NAMES,
            y=p_base,
            name="P(ready), typical",
            line=dict(color=ESTATE, width=3),
            yaxis="y2",
        )
    )
    if rain_change or temp_change:
        fig.add_trace(
            go.Scatter(
                x=MONTH_NAMES,
                y=p_scen,
                name="P(ready), scenario",
                line=dict(color=LIQUOR, width=3, dash="dash"),
                yaxis="y2",
            )
        )
    fig.update_layout(
        yaxis=dict(title="Typical rainfall (mm)", side="right", showgrid=False),
        yaxis2=dict(
            title="P(ready)",
            range=[0, 1.05],
            tickformat=".0%",
            dtick=0.25,
            overlaying="y",
            side="left",
        ),
        legend=dict(orientation="h", y=-0.2),
    )
    st.plotly_chart(plotly_layout(fig, height=380), width="stretch")
    table = base[
        ["month", "rainfall_mm", "temperature_c", "humidity_percent", "soil_moisture_percent"]
    ].copy()
    table["month"] = MONTH_NAMES
    table["P(ready) typical"] = p_base.round(3)
    table["P(ready) scenario"] = p_scen.round(3)
    st.download_button(
        "Download the calendar (CSV)",
        table.to_csv(index=False).encode(),
        file_name=f"plucking_calendar_{region.replace(' ', '_')}.csv",
        mime="text/csv",
    )


def page_batch():
    st.title("Score many months")
    st.markdown(
        '<p class="lede">Upload a CSV with one row per region-month. The app checks each row, '
        "predicts readiness and lets you download the results.</p>",
        unsafe_allow_html=True,
    )
    if not require_model():
        return
    template = pd.DataFrame(
        [
            {
                "region": "Nuwara Eliya",
                "year": 2025,
                "month": 1,
                "rainfall_mm": 120.0,
                "temperature_c": 14.5,
                "humidity_percent": 84.0,
                "soil_moisture_percent": 68.0,
            },
            {
                "region": "Galle",
                "year": 2025,
                "month": 4,
                "rainfall_mm": 260.0,
                "temperature_c": 27.8,
                "humidity_percent": 86.0,
                "soil_moisture_percent": 79.0,
            },
        ]
    )
    st.download_button(
        "Download a template CSV",
        template.to_csv(index=False).encode(),
        file_name="tea_batch_template.csv",
        mime="text/csv",
    )
    st.caption(
        "Required columns: region, month, rainfall_mm, temperature_c, humidity_percent, "
        "soil_moisture_percent (year is optional). Regions must be one of: "
        + ", ".join(REGIONS)
        + "."
    )
    up = st.file_uploader("CSV file", type=["csv"])
    if up is None:
        return
    try:
        data = pd.read_csv(io.BytesIO(up.getvalue()))
    except Exception as err:
        st.error(
            f"This file could not be read as CSV ({type(err).__name__}). Save it as comma-separated UTF-8."
        )
        return
    required = [
        "region",
        "month",
        "rainfall_mm",
        "temperature_c",
        "humidity_percent",
        "soil_moisture_percent",
    ]
    missing = [c for c in required if c not in data.columns]
    if missing:
        st.error(f"Missing column(s): {', '.join(missing)}. Use the template above.")
        return
    if len(data) > 5000:
        st.error(f"The file has {len(data):,} rows; the limit is 5,000 per upload.")
        return

    data = data.copy()
    data["region"] = data["region"].astype(str).str.strip()
    status = []
    for _, r in data.iterrows():
        issues = []
        if r["region"] not in REGIONS:
            issues.append(f"unknown region '{r['region']}'")
        issues += validate_inputs(r.to_dict())
        status.append("; ".join(issues))
    data["issues"] = status
    valid = data["issues"] == ""
    data["elevation_m"] = data["region"].map(lambda x: REGIONS.get(x, {}).get("elevation_m"))
    data["probability_ready"] = np.nan
    if valid.any():
        data.loc[valid, "probability_ready"] = predict_local(data[valid]).round(4)
    data["prediction"] = np.where(
        data["probability_ready"].isna(),
        "invalid row",
        np.where(data["probability_ready"] >= THRESHOLD, "ready", "not ready"),
    )
    a, b, c = st.columns(3)
    a.metric("Rows", len(data))
    b.metric("Predicted ready", int((data.prediction == "ready").sum()))
    c.metric("Rows with problems", int((~valid).sum()))
    if (~valid).any():
        st.warning("Some rows were skipped. See the 'issues' column.")
    st.dataframe(
        data,
        width="stretch",
        hide_index=True,
        column_config={
            "probability_ready": st.column_config.ProgressColumn(
                "P(ready)", min_value=0.0, max_value=1.0, format="%.2f"
            )
        },
    )
    st.download_button(
        "Download results (CSV)",
        data.to_csv(index=False).encode(),
        file_name="tea_batch_predictions.csv",
        mime="text/csv",
    )


def page_explore():
    st.title("Explore the data")
    if not require_data():
        return
    c1, c2 = st.columns([2, 1])
    regions = c1.multiselect("Regions", list(REGIONS), default=list(REGIONS))
    years = c2.slider("Years", 2001, 2024, (2001, 2024))
    d = df[df.region.isin(regions) & df.year.between(*years)]
    if d.empty:
        st.info("No rows match. Select at least one region.")
        return
    a, b, c, e = st.columns(4)
    a.metric("Rows", f"{len(d):,}")
    b.metric("Harvest-ready", f"{d.harvest_ready.mean():.0%}")
    c.metric("Median rainfall", f"{d.rainfall_mm.median():.0f} mm")
    e.metric("Median temperature", f"{d.temperature_c.median():.1f} °C")

    order = [r for r in REGIONS if r in regions]
    heat = d.pivot_table(index="region", columns="month", values="harvest_ready").reindex(order)
    fig = go.Figure(
        go.Heatmap(
            z=heat.values,
            x=MONTH_NAMES,
            y=heat.index,
            zmin=0,
            zmax=1,
            colorscale=[[0, LIQUOR], [0.5, "#F1EBC8"], [1, LEAF]],
            text=np.round(heat.values * 100).astype(int),
            texttemplate="%{text}%",
            hovertemplate="%{y}, %{x}: %{z:.0%} ready<extra></extra>",
            colorbar=dict(title="Ready", tickformat=".0%"),
        )
    )
    fig.update_yaxes(autorange="reversed")  # first region (highest elevation) at the top
    st.plotly_chart(
        plotly_layout(
            fig,
            height=120 + 34 * len(order),
            title="Harvest-ready rate by region and month (highest elevation at top)",
        ),
        width="stretch",
    )

    feat = st.selectbox(
        "Compare a weather variable between ready and not-ready months",
        ["rainfall_mm", "temperature_c", "humidity_percent", "soil_moisture_percent"],
        format_func=lambda f: FEATURE_LABELS[f],
    )
    fig = go.Figure()
    for cls, name, color in [(0, "Not ready", LIQUOR), (1, "Ready", LEAF)]:
        fig.add_trace(
            go.Histogram(
                x=d.loc[d.harvest_ready == cls, feat],
                name=name,
                marker_color=color,
                opacity=0.7,
                nbinsx=40,
            )
        )
    fig.update_layout(barmode="overlay", legend=dict(orientation="h", y=1.1))
    st.plotly_chart(
        plotly_layout(fig, height=330, xaxis_title=FEATURE_LABELS[feat], yaxis_title="Months"),
        width="stretch",
    )

    trend = (
        d.groupby("year")
        .agg(ready=("harvest_ready", "mean"), rain=("rainfall_mm", "mean"))
        .reset_index()
    )
    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=trend.year,
            y=trend.rain,
            name="Mean monthly rainfall",
            marker_color="#B7CDB2",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=trend.year,
            y=trend.ready,
            name="Ready rate",
            line=dict(color=ESTATE, width=3),
            yaxis="y2",
        )
    )
    fig.update_layout(
        yaxis=dict(side="right", title="Rainfall (mm)", showgrid=False),
        yaxis2=dict(overlaying="y", side="left", tickformat=".0%", title="Ready rate"),
        legend=dict(orientation="h", y=1.12),
    )
    st.plotly_chart(plotly_layout(fig, height=320, title="Year-by-year"), width="stretch")
    with st.expander("Show the filtered rows"):
        st.dataframe(d.drop(columns="month_name"), width="stretch", hide_index=True)


def page_models():
    st.title("How the models compare")
    st.markdown(
        '<p class="lede">Nine approaches were tuned and compared. The Explainable Boosting Machine '
        "performed on par with the black-box ensembles (within about one percentage point, with no "
        "statistically significant difference) while staying fully interpretable, so it powers this app.</p>",
        unsafe_allow_html=True,
    )
    test = load_csv("model_comparison_test.csv")
    val = load_csv("model_comparison_validation.csv")
    if test is None:
        st.warning("Metrics files not found in reports/metrics. Run the notebook to create them.")
        return
    tab1, tab2, tab3 = st.tabs(["Test set", "Validation set and tuning", "Is 99% real?"])
    with tab1:
        cols = [
            "Accuracy",
            "F1",
            "ROC-AUC",
            "PR-AUC",
            "MCC",
            "Recall (ready)",
            "Recall (not ready)",
        ]
        show = test[[c for c in cols if c in test.columns]]
        st.dataframe(
            show.style.format("{:.4f}").highlight_max(color="#D8EBD3"),
            width="stretch",
        )
        d = show.drop(index="Majority baseline", errors="ignore").sort_values("F1")
        fig = go.Figure(
            go.Bar(
                x=d["F1"],
                y=d.index,
                orientation="h",
                text=[f"{v:.3f}" for v in d["F1"]],
                textposition="outside",
                marker_color=[LEAF if "EBM" in i else "#9DB89A" for i in d.index],
            )
        )
        fig.update_xaxes(range=[max(0, d["F1"].min() - 0.05), 1.01])
        st.plotly_chart(
            plotly_layout(fig, height=360, title="Test F1 by model (EBM in dark green)"),
            width="stretch",
        )
        roc = load_json("roc_curves_test.json")
        if roc:
            fig = go.Figure()
            for name, curve in roc.items():
                fig.add_trace(
                    go.Scatter(
                        x=curve["fpr"],
                        y=curve["tpr"],
                        name=name,
                        mode="lines",
                        line=dict(
                            width=3.5 if "EBM" in name else 1.5,
                            color=LEAF if "EBM" in name else None,
                        ),
                    )
                )
            fig.add_trace(
                go.Scatter(
                    x=[0, 1],
                    y=[0, 1],
                    mode="lines",
                    line=dict(dash="dash", color="#9AA8A0"),
                    showlegend=False,
                )
            )
            st.plotly_chart(
                plotly_layout(
                    fig,
                    height=420,
                    title="ROC curves on the test set",
                    xaxis_title="False positive rate",
                    yaxis_title="True positive rate",
                ),
                width="stretch",
            )
    with tab2:
        if val is not None:
            st.dataframe(val.style.format("{:.4f}"), width="stretch")
        hp = load_csv("hyperparameter_search.csv")
        if hp is not None:
            st.markdown("**Best hyperparameters found by randomized search**")
            st.dataframe(hp, width="stretch")
        cv = load_csv("cv_scores.csv")
        if cv is not None:
            st.markdown("**Cross-validated F1 on the training set**")
            st.dataframe(cv, width="stretch")
    with tab3:
        checks = load_json("validation_checks.json") or {}
        if checks:
            ts = checks.get("time_split_2001_2019_vs_2020_2024", {})
            lo = checks.get("leave_one_region_out", {})
            pt = checks.get("permutation_test", {})
            a, b, c, e = st.columns(4)
            a.metric("Trained ≤2019, tested 2020–24", f"{ts.get('Accuracy', 0):.1%}")
            b.metric(
                "Unseen region (mean)",
                f"{lo.get('mean_accuracy', 0):.1%}",
                help=f"Worst region: {lo.get('min_accuracy', 0):.1%}",
            )
            c.metric(
                "AUC with shuffled labels",
                f"{pt.get('shuffled_auc_mean', 0):.2f}",
                help="≈0.5 means no target leakage",
            )
            e.metric(
                "Depth-3 rule tree", f"{checks.get('depth3_tree_on_weather_val_accuracy', 0):.1%}"
            )
            st.markdown(
                "The near-perfect score holds on future years and on regions the model never saw, and it collapses "
                "to chance when the labels are shuffled, so there is no leakage. A tiny three-level tree on four "
                "weather columns already gets most of the way, which shows the readiness label follows agronomic "
                "weather thresholds. The model learned those thresholds; forecasting next month is harder:"
            )
        nm = load_csv("next_month_forecast.csv")
        if nm is not None:
            st.dataframe(nm.style.format("{:.4f}"), width="stretch")
        for name in ["validation_rule_tree", "validation_learning_curve"]:
            p = FIG_DIR / f"{name}.png"
            if p.exists():
                st.image(str(p), width="stretch")


def page_explain():
    st.title("What the model has learned")
    st.markdown(
        '<p class="lede">An Explainable Boosting Machine adds up one curve per weather factor (plus a few '
        "pairwise interactions). These curves are the model itself, so what you see here is exactly how "
        "it decides.</p>",
        unsafe_allow_html=True,
    )
    if not require_model():
        return
    ebm = model.named_steps["model"]
    imp = pd.Series(
        ebm.term_importances(), index=[pretty_term(t) for t in ebm.term_names_]
    ).sort_values()
    fig = go.Figure(
        go.Bar(
            x=imp.values,
            y=imp.index,
            orientation="h",
            marker_color=[LIQUOR if "×" in t else LEAF for t in imp.index],
        )
    )
    st.plotly_chart(
        plotly_layout(
            fig,
            height=60 + 26 * len(imp),
            title="Importance of each term (green = single factor, gold = interaction)",
            xaxis_title="Mean |contribution| to log-odds",
        ),
        width="stretch",
    )

    feat = st.selectbox(
        "Show the learned curve for",
        MODEL_FEATURES,
        index=MODEL_FEATURES.index("rainfall_mm"),
        format_func=lambda f: FEATURE_LABELS[f],
    )
    try:
        data = ebm.explain_global().data(ebm.term_names_.index(feat))
        edges = np.asarray(data["names"], dtype=float)
        scores = np.asarray(data["scores"], dtype=float)
        x = (edges[:-1] + edges[1:]) / 2 if len(edges) == len(scores) + 1 else edges
        lo = np.asarray(data.get("lower_bounds", scores), dtype=float)
        hi = np.asarray(data.get("upper_bounds", scores), dtype=float)
        lo_x, hi_x = TRAINING_RANGES[feat]
        keep = (x >= lo_x - 1) & (x <= hi_x + 1)
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=np.r_[x[keep], x[keep][::-1]],
                y=np.r_[hi[keep], lo[keep][::-1]],
                fill="toself",
                fillcolor="rgba(62,124,58,0.18)",
                line=dict(width=0),
                hoverinfo="skip",
                name="Uncertainty",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=x[keep],
                y=scores[keep],
                line=dict(color=ESTATE, width=3, shape="hv"),
                name="Contribution",
                hovertemplate="%{x:.1f} → %{y:+.2f}<extra></extra>",
            )
        )
        fig.add_hline(y=0, line_dash="dash", line_color=MUTED)
        st.plotly_chart(
            plotly_layout(
                fig,
                height=380,
                showlegend=False,
                xaxis_title=FEATURE_LABELS[feat],
                yaxis_title="Contribution to log-odds of ready",
                title=f"Learned effect of {FEATURE_LABELS[feat].lower()}",
            ),
            width="stretch",
        )
        st.caption(
            "Above the dashed line the factor pushes towards ready; below, towards not ready. "
            "The shaded band shows how much the curve varied across the model's bootstrap bags."
        )
    except Exception as err:
        st.error(f"Could not draw this curve: {err}")

    st.subheader("Post-hoc explanations from the notebook")
    st.markdown(
        "SHAP and LIME were used to cross-check the model's own explanation. All three agree that "
        "rainfall, temperature and humidity drive readiness, with thresholds near 80 mm of rain and "
        "90% humidity."
    )
    tabs = st.tabs(
        [
            "SHAP summary",
            "SHAP dependence",
            "SHAP waterfall (one month)",
            "LIME (same month)",
            "Partial dependence",
        ]
    )
    for tab, name in zip(
        tabs,
        ["shap_beeswarm", "shap_dependence", "shap_waterfall_heldout", "lime_heldout", "pdp_ice"],
    ):
        with tab:
            p = FIG_DIR / f"{name}.png"
            if p.exists():
                st.image(str(p), width="stretch")
            else:
                st.info(f"{name}.png not found. Run the notebook to generate it.")


def page_about():
    st.title("About this project")
    st.markdown("""
**Tea Flush Planner** is my end-to-end applied machine learning project for the MSc in Artificial Intelligence
(University of Moratuwa), Applied Machine Learning module.

**Problem.** Estates must send plucking gangs out when the flush is ready. Sending them too early wastes wages;
too late, and the leaf over-matures and loses grade.

**Data.** 3,168 region-months from 11 Sri Lankan tea regions (2001–2024): rainfall, temperature, humidity,
soil moisture, elevation and a harvest-readiness label. No personal data is involved.

**Model.** An Explainable Boosting Machine (a glass-box generalised additive model with pairwise interactions),
compared against logistic regression, a decision tree, Random Forest, XGBoost, LightGBM, CatBoost and stacking.

**Explainability.** EBM shape functions, SHAP, LIME, partial dependence and a counterfactual search.

**MLOps.** Data and model versioned with DVC on DagsHub, experiments and the model registry in MLflow on DagsHub,
tests and a champion/challenger quality gate in GitHub Actions, a FastAPI service and this app in Docker on Azure
Container Apps with canary releases, drift monitoring and automated retraining.

**Limits.** The label follows weather rules, the weather is gridded and monthly, and the model cannot see pests,
fertiliser or labour. Use it to plan, then confirm in the field.
""")
    card = load_model_card()
    st.caption(f"Model file: {model_source}" + (f". Live API: {API_URL}" if API_URL else ""))
    if card:
        with st.expander("Model card"):
            st.json(card)


# ===========================================================================
# Navigation
# ===========================================================================
LOGO = Path(__file__).resolve().parent / "assets" / "logo.svg"
if LOGO.exists():
    st.logo(str(LOGO), size="large")

pages = [
    st.Page(page_home, title="Overview", icon=":material/eco:", default=True),
    st.Page(
        page_predict,
        title="Predict a month",
        icon=":material/online_prediction:",
        url_path="predict",
    ),
    st.Page(page_whatif, title="What-if simulator", icon=":material/tune:", url_path="what-if"),
    st.Page(
        page_calendar,
        title="Plucking calendar",
        icon=":material/calendar_month:",
        url_path="calendar",
    ),
    st.Page(page_batch, title="Score many months", icon=":material/upload_file:", url_path="batch"),
    st.Page(page_explore, title="Explore the data", icon=":material/insights:", url_path="explore"),
    st.Page(
        page_models, title="Model comparison", icon=":material/leaderboard:", url_path="models"
    ),
    st.Page(
        page_explain,
        title="What the model learned",
        icon=":material/psychology:",
        url_path="explain",
    ),
    st.Page(page_about, title="About", icon=":material/info:", url_path="about"),
]
nav = st.navigation(pages)

with st.sidebar:
    st.divider()
    if model is None:
        st.error("Model not loaded")
    else:
        st.caption(f"Model: Explainable Boosting Machine\n\nSource: {model_source}")
    if API_URL:
        st.caption(f"Live API: {API_URL}")
    st.caption("Built by Thejina, MSc in AI, University of Moratuwa")

nav.run()
