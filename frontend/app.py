import os
import html
import streamlit as st
import streamlit.components.v1 as components
import requests
import pandas as pd

from content import (
    EXAMPLE_PATIENTS,
    GLOSSARY,
    HOW_TO_USE,
    DISCLAIMER,
    ABOUT_ANALYSIS,
    MAGNITUDE_LABELS,
)

st.set_page_config(
    page_title="Causal Pharma Intelligence",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded",
)

_BASE_DIR = os.path.dirname(os.path.abspath(__file__))
BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")


def load_css():
    css_path = os.path.join(_BASE_DIR, "assets", "style.css")
    if os.path.exists(css_path):
        with open(css_path) as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)


load_css()


# Reference DAG used when /dag/reference is unavailable (older backend)
REFERENCE_DAG_FALLBACK = {
    "nodes": ["age", "blood_pressure", "comorbidities", "treatment", "outcome"],
    "edges": [
        {"source": "age", "target": "treatment"},
        {"source": "blood_pressure", "target": "treatment"},
        {"source": "comorbidities", "target": "treatment"},
        {"source": "age", "target": "outcome"},
        {"source": "blood_pressure", "target": "outcome"},
        {"source": "comorbidities", "target": "outcome"},
        {"source": "treatment", "target": "outcome"},
    ],
}


def call_api(method: str, endpoint: str, payload=None, silent=False):
    url = f"{BACKEND_URL.rstrip('/')}{endpoint}"
    try:
        if method == "GET":
            r = requests.get(url, timeout=60)
        else:
            r = requests.post(url, json=payload, timeout=60)
        r.raise_for_status()
        return r.json()
    except requests.exceptions.ConnectionError:
        if not silent:
            message_box(
                f"Cannot connect to backend at <code>{html.escape(url)}</code>. "
                "Run <code>docker-compose up --build</code> and wait for the API.",
                "error",
            )
    except requests.exceptions.Timeout:
        if not silent:
            message_box("Request timed out. SHAP may take ~30s — try again.", "warn")
    except requests.exceptions.HTTPError as e:
        if not silent:
            message_box(
                f"API error {e.response.status_code} for <code>{endpoint}</code> — "
                "restart backend: <code>docker-compose restart backend</code>",
                "error",
            )
    except Exception as e:
        if not silent:
            message_box(f"Unexpected error: {html.escape(str(e))}", "error")
    return None


def simulate_both_fallback(payload, cate_val):
    """Fallback when /simulate_both is missing on an older backend."""
    y0 = call_api("POST", "/simulate_intervention", {**payload, "treatment_value": 0}, silent=True)
    y1 = call_api("POST", "/simulate_intervention", {**payload, "treatment_value": 1}, silent=True)
    if y0 and y1:
        return {
            "outcome_if_withhold": y0.get("expected_outcomes", []),
            "outcome_if_treat": y1.get("expected_outcomes", []),
            "treatment_effect": [cate_val],
        }
    return None


# ── UI helpers (no Plotly/iframes — avoids pink empty chart boxes) ─────────────

def message_box(text: str, variant: str = "info"):
    """Custom alert box — replaces st.info/success/warning (no pink Streamlit alerts)."""
    styles = {
        "info": ("#eff6ff", "#3b82f6"),
        "ok": ("#ecfdf5", "#059669"),
        "warn": ("#fffbeb", "#d97706"),
        "error": ("#fef2f2", "#dc2626"),
    }
    bg, border = styles.get(variant, styles["info"])
    st.markdown(
        f'<div class="msg-box" style="background:{bg};border-left:4px solid {border};'
        f'padding:14px 18px;border-radius:8px;color:#0f172a;margin:12px 0;line-height:1.6;">'
        f"{text}</div>",
        unsafe_allow_html=True,
    )


def _format_node_label(name: str) -> str:
    return name.replace("_", " ").title()


def _mermaid_safe(name: str) -> str:
    return name.replace(" ", "_").replace("-", "_")


def render_dag_mermaid(nodes, edges, title: str, subtitle: str = ""):
    """DAG via Mermaid (renders in-page, no iframe)."""
    st.markdown(f"**{title}**")
    if subtitle:
        st.caption(subtitle)
    lines = ["flowchart LR"]
    linked = set()
    for e in edges:
        s = _mermaid_safe(e["source"])
        t = _mermaid_safe(e["target"])
        sl = _format_node_label(e["source"])
        tl = _format_node_label(e["target"])
        lines.append(f'    {s}["{sl}"] --> {t}["{tl}"]')
        linked.add(s)
        linked.add(t)
    for n in nodes:
        mid = _mermaid_safe(n)
        if mid not in linked:
            lines.append(f'    {mid}["{_format_node_label(n)}"]')
    mermaid_code = "\n".join(lines)
    html_code = f"""
    <div class="mermaid" style="display: flex; justify-content: center;">
{mermaid_code}
    </div>
    <script src="https://cdn.jsdelivr.net/npm/mermaid/dist/mermaid.min.js"></script>
    <script>
        mermaid.initialize({{startOnLoad: true}});
    </script>
    """
    components.html(html_code, height=550)


def render_horizontal_bars(rows, title: str = ""):
    """
    HTML horizontal bars. rows = [(label, value), ...]
    No iframes — always visible.
    """
    if title:
        st.markdown(f"**{title}**")
    max_abs = max(abs(v) for _, v in rows) or 1.0
    parts = ['<div class="bar-chart">']
    for label, value in rows:
        pct = max(4, int(100 * abs(value) / max_abs))
        color = "#059669" if value >= 0 else "#dc2626"
        parts.append(
            f'<div class="bar-row">'
            f'<span class="bar-label">{html.escape(label)}</span>'
            f'<div class="bar-track"><div class="bar-fill" style="width:{pct}%;background:{color};"></div></div>'
            f'<span class="bar-value">{value:+.3f}</span></div>'
        )
    parts.append("</div>")
    st.markdown("".join(parts), unsafe_allow_html=True)


def draw_counterfactual(y0, y1, cate_val):
    diff = y1 - y0
    c1, c2, c3 = st.columns(3)
    c1.metric("If we withhold (T=0)", f"{y0:.1f}")
    c2.metric("If we treat (T=1)", f"{y1:.1f}")
    c3.metric("Difference", f"{diff:+.1f}")
    render_horizontal_bars(
        [("Withhold (T=0)", y0), ("Treat (T=1)", y1)],
        title="Expected outcome comparison",
    )
    sign = "more" if diff > 0 else "less"
    message_box(
        f"<strong>Treating vs withholding:</strong> expected outcome is "
        f"<strong>{abs(diff):.1f} units {sign}</strong> with treatment "
        f"(individual effect ≈ {cate_val:+.2f}).",
        "info",
    )


def draw_shap_bars(feature_labels, shap_values, base_value: float):
    render_horizontal_bars(
        list(zip(feature_labels, shap_values)),
        title=f"Why this CATE? (baseline = {base_value:+.3f})",
    )
    st.caption("Green = pushes treatment effect up · Red = pushes it down")


def clinical_summary(age, bp, comorbidities, cate_val, ate_val, rec_label, magnitude, ci_lo, ci_hi):
    direction = "improve" if cate_val > 0 else "worsen" if cate_val < 0 else "not materially change"
    mag_text = MAGNITUDE_LABELS.get(magnitude, "Unknown impact")
    message_box(
        f"<strong>Clinical summary</strong> — For a <strong>{int(age)}-year-old</strong> patient with "
        f"<strong>BP {int(bp)} mmHg</strong> and "
        f"<strong>{comorbidities} comorbidit{'y' if comorbidities == 1 else 'ies'}</strong>, "
        f"treatment is estimated to <strong>{direction}</strong> the outcome by "
        f"<strong>{abs(cate_val):.2f} units</strong> ({mag_text.lower()}). "
        f"95% interval: <strong>[{ci_lo:+.2f}, {ci_hi:+.2f}]</strong>. "
        f"<strong>Recommended: {rec_label}</strong>. ATE: <strong>{ate_val:+.2f}</strong>.",
        "ok",
    )


# ── Session defaults ──────────────────────────────────────────────────────────

if "age" not in st.session_state:
    st.session_state.age = 65
if "bp" not in st.session_state:
    st.session_state.bp = 120
if "comorbidities" not in st.session_state:
    st.session_state.comorbidities = 2

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class='page-header'>
    <div class='page-title'>Causal Pharma Intelligence</div>
    <div class='page-subtitle'>Understand treatment effects — step by step</div>
</div>
""", unsafe_allow_html=True)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("<div class='sidebar-section'>Example patient</div>", unsafe_allow_html=True)
    preset = st.selectbox("Load profile", list(EXAMPLE_PATIENTS.keys()), label_visibility="collapsed")
    if EXAMPLE_PATIENTS[preset] is not None:
        p = EXAMPLE_PATIENTS[preset]
        st.session_state.age = p["age"]
        st.session_state.bp = p["blood_pressure"]
        st.session_state.comorbidities = p["comorbidities"]

    st.markdown("<div class='sidebar-section'>Patient parameters</div>", unsafe_allow_html=True)
    age = st.slider("Age (years)", 18, 95, st.session_state.age, key="age")
    bp = st.slider("Blood pressure (mmHg)", 80, 200, st.session_state.bp, key="bp")
    comorbidities = int(st.number_input("Comorbidities", 0, 10, st.session_state.comorbidities, key="comorbidities"))

    st.markdown("<div class='sidebar-divider'></div>", unsafe_allow_html=True)
    with st.expander("Glossary"):
        for term, definition in GLOSSARY.items():
            st.markdown(f"**{term}** — {definition}")

    with st.expander("About this analysis"):
        st.markdown(ABOUT_ANALYSIS)

    st.markdown("<div class='sidebar-divider'></div>", unsafe_allow_html=True)
    run_analysis = st.button("Run Analysis", use_container_width=True, type="primary")

# ── Run analysis ──────────────────────────────────────────────────────────────
if run_analysis:
    payload = {"features": [{"age": float(age), "blood_pressure": float(bp), "comorbidities": comorbidities}]}
    api_errors = []
    with st.spinner("Computing causal estimates and explanations (SHAP may take ~30s)..."):
        cate = call_api("POST", "/predict_cate", payload, silent=True)
        policy = call_api("POST", "/policy_decision", payload, silent=True)
        dag = call_api("GET", "/dag", silent=True)
        dag_ref = call_api("GET", "/dag/reference", silent=True)
        sim_both = call_api("POST", "/simulate_both", payload, silent=True)
        explain = call_api("POST", "/explain_prediction", payload, silent=True)

        if not cate:
            api_errors.append("`/predict_cate` — is the backend running on port 8000?")
        if not policy:
            api_errors.append("`/policy_decision`")

        cate_val_tmp = (cate or {}).get("cate_estimates", [0.0])[0]
        if not sim_both:
            sim_both = simulate_both_fallback(payload, cate_val_tmp)
            if not sim_both:
                api_errors.append("`/simulate_both` (restart backend to enable)")
        if not dag_ref:
            dag_ref = REFERENCE_DAG_FALLBACK
        if not explain:
            api_errors.append("`/explain_prediction` (restart backend to enable SHAP)")

        st.session_state.results = {
            "cate": cate,
            "policy": policy,
            "dag": dag,
            "dag_ref": dag_ref,
            "sim_both": sim_both,
            "explain": explain,
            "age": age, "bp": bp, "comorbidities": comorbidities,
            "api_errors": api_errors,
        }

    if api_errors and not cate:
        message_box(
            "Analysis failed. Start the stack with <code>docker-compose up --build</code>, "
            "wait until http://localhost:8000/health returns OK, then run again.<br><br>"
            + "<br>".join(f"• {html.escape(e)}" for e in api_errors),
            "error",
        )
    elif api_errors:
        message_box(
            "Some features need a backend restart: <code>docker-compose restart backend</code><br><br>"
            + "<br>".join(f"• {html.escape(e)}" for e in api_errors),
            "warn",
        )

if "results" in st.session_state and st.session_state.results.get("cate") and st.session_state.results.get("policy"):
    R = st.session_state.results
    cate_resp = R["cate"]
    policy_resp = R["policy"]
    cate_val = cate_resp["cate_estimates"][0]
    ate_val = cate_resp["ate_estimate"]
    margin = max(0.3, abs(cate_val) * 0.15)
    ci_lo = cate_resp.get("ci_lower", [cate_val - margin])[0]
    ci_hi = cate_resp.get("ci_upper", [cate_val + margin])[0]
    magnitude = cate_resp.get("magnitude", ["moderate" if abs(cate_val) >= 2 else "small"])[0]
    confidence = cate_resp.get("confidence_pct", [80])[0]
    rec = policy_resp["recommended_treatment"][0]
    rec_label = "Treat (T=1)" if rec == 1 else "Withhold (T=0)"
    rationale = policy_resp.get("rationale", [
        f"Estimated treatment effect is {cate_val:+.2f} outcome units for this patient."
    ])[0]
    bandit_rec = policy_resp.get("bandit_recommendation", [rec])[0]
    bandit_label = "Treat (T=1)" if bandit_rec == 1 else "Withhold (T=0)"

    sim = R.get("sim_both") or {}
    y0 = sim.get("outcome_if_withhold", [None])[0]
    y1 = sim.get("outcome_if_treat", [None])[0]

    clinical_summary(R["age"], R["bp"], R["comorbidities"], cate_val, ate_val, rec_label, magnitude, ci_lo, ci_hi)

    # Radio navigation — only renders active section (fixes empty pink boxes in hidden tabs)
    view = st.radio(
        "Step",
        ["① Recommendation", "② Why?", "③ What-if", "④ Details"],
        horizontal=True,
        label_visibility="collapsed",
    )

    if view == "① Recommendation":
        k1, k2, k3, k4 = st.columns(4)
        with k1:
            st.metric("ATE (avg. effect)", f"{ate_val:+.3f}", help="Average benefit across similar patients in training data")
        with k2:
            st.metric(
                "CATE (this patient)",
                f"{cate_val:+.3f}",
                delta=f"95% CI [{ci_lo:+.2f}, {ci_hi:+.2f}]",
                delta_color="normal",
                help="Expected benefit for this specific patient",
            )
        with k3:
            st.metric("Policy", rec_label, help="Recommended treatment based on CATE sign")
        with k4:
            st.metric("Impact", MAGNITUDE_LABELS[magnitude], delta=f"~{confidence}% confidence", delta_color="off")

        message_box(html.escape(rationale), "info")
        if bandit_rec != rec:
            message_box(
                f"LinUCB bandit (exploratory) suggests <strong>{bandit_label}</strong> "
                "— may differ when exploring trade-offs.",
                "warn",
            )
        else:
            st.caption(f"LinUCB bandit agrees with the CATE-based policy ({bandit_label}).")

        with st.expander("What do these numbers mean?"):
            for term, definition in GLOSSARY.items():
                st.markdown(f"**{term}** — {definition}")

    elif view == "② Why?":
        st.caption("Causal links from data (left) vs clinical reference (right).")
        col_l, col_r = st.columns(2)
        with col_l:
            if R.get("dag"):
                n_edges = len(R["dag"].get("edges", []))
                render_dag_mermaid(
                    R["dag"]["nodes"],
                    R["dag"]["edges"],
                    "Learned from data",
                    f"{n_edges} edge(s) from NOTEARS" if n_edges else "No strong edges at current threshold",
                )
        with col_r:
            if R.get("dag_ref"):
                render_dag_mermaid(
                    R["dag_ref"]["nodes"],
                    R["dag_ref"]["edges"],
                    "Clinical reference",
                    "Expected relationships from domain knowledge",
                )

        st.subheader("Feature drivers (SHAP)")
        explain = R.get("explain")
        if explain and explain.get("shap_values"):
            labels = explain.get("feature_labels", explain["feature_names"])
            shap_row = explain["shap_values"][0]
            draw_shap_bars(labels, shap_row, explain["base_value"])
        else:
            message_box(
                "SHAP explanation unavailable. Restart backend: "
                "<code>docker-compose restart backend</code>",
                "warn",
            )

    elif view == "③ What-if":
        if y0 is not None and y1 is not None:
            draw_counterfactual(y0, y1, cate_val)
        else:
            message_box("Counterfactual outcomes unavailable.", "warn")

    elif view == "④ Details":
        st.subheader("Patient summary")
        patient_id = f"P-{int(R['age']):03d}{int(R['bp']):03d}{R['comorbidities']}"
        d1, d2, d3, d4 = st.columns(4)
        d1.metric("Patient ID", patient_id)
        d2.metric("Age", f"{int(R['age'])} yrs")
        d3.metric("Blood pressure", f"{int(R['bp'])} mmHg")
        d4.metric("Comorbidities", R["comorbidities"])

        st.subheader("Treatment analysis")
        t1, t2, t3, t4 = st.columns(4)
        t1.metric("ATE", f"{ate_val:+.3f}")
        t2.metric("CATE", f"{cate_val:+.3f}")
        t3.metric("Policy", rec_label)
        t4.metric("Confidence", f"{confidence}%")

        if y0 is not None and y1 is not None:
            st.subheader("Outcome comparison")
            st.table(
                pd.DataFrame({
                    "Scenario": ["Control (T=0)", "Treated (T=1)"],
                    "Expected outcome": [f"{y0:.1f}", f"{y1:.1f}"],
                    "vs control": ["—", f"{y1 - y0:+.1f}"],
                })
            )

else:
    message_box(
        "<strong>Ready for analysis</strong> — Choose an <strong>example patient</strong> "
        "in the sidebar (or adjust sliders), then click <strong>Run Analysis</strong>.",
        "info",
    )
    st.markdown(
        f"**Current patient:** age **{int(age)}**, BP **{int(bp)}**, "
        f"comorbidities **{comorbidities}**"
    )
