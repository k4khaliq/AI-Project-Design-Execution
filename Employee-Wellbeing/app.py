# app.py

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

from dotenv import load_dotenv

from config import DEFAULT_DATA_VERSION, DATA_DIR
from model_training import train_pipeline, BurnoutPredictor
from data_generation import EmployeeDataConfig, generate_and_save
from rag_engine import HRPolicyRAGEngine
from llm_integration import LLMClient, LLMConfig

load_dotenv()

# ---------------- Streamlit base config ----------------

st.set_page_config(
    page_title="Well-Being System",
    page_icon="🧠",
    layout="wide",
)

# ---------------- CSS polish ----------------

st.markdown(
    """
    <style>
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0f172a 0%, #111827 40%, #1f2937 100%);
        color: #e5e7eb !important;
    }
    [data-testid="stSidebar"] * {
        color: #e5e7eb !important;
    }
    .stButton>button {
        border-radius: 999px;
        padding: 0.5rem 1.4rem;
        border: none;
        font-weight: 600;
    }
    .kpi-card {
        border-radius: 0.9rem;
        padding: 1rem 1.4rem;
        background: #ffffff;
        box-shadow: 0 8px 20px rgba(15,23,42,0.06);
        border: 1px solid #e5e7eb;
    }
    .kpi-label {
        font-size: 0.9rem;
        color: #6b7280;
        text-transform: uppercase;
        letter-spacing: .06em;
        margin-bottom: 0.35rem;
    }
    .kpi-value {
        font-size: 2.0rem;
        font-weight: 700;
        color: #111827;
    }
    .kpi-sub {
        font-size: 0.85rem;
        color: #6b7280;
        margin-top: 0.15rem;
    }
    .section-title {
        font-size: 1.3rem;
        font-weight: 700;
        margin-bottom: 0.35rem;
        color: #111827;
    }
    .section-subtitle {
        font-size: 0.9rem;
        color: #6b7280;
        margin-bottom: 0.75rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------- Data + model loading ----------------

@st.cache_data
def load_employee_data(version: str = DEFAULT_DATA_VERSION) -> pd.DataFrame:
    path = DATA_DIR / f"employees_{version}.parquet"
    if not path.exists():
        # generate 5000 synthetic employees by default
        generate_and_save(EmployeeDataConfig(n_employees=5000, version=version), path)
    df = pd.read_parquet(path)

    # Ensure Department exists
    if "Department" not in df.columns:
        np.random.seed(42)
        departments = [
            "Engineering", "Data Science", "IT", "HR", "Finance", "Customer Service",
            "Marketing", "Sales", "Operations", "Product", "Legal", "R&D",
            "Quality Assurance", "Supply Chain", "Business Development",
        ]
        df["Department"] = np.random.choice(departments, size=len(df))

    # Ensure EmployeeName exists (synthetic but realistic)
    if "EmployeeName" not in df.columns:
        np.random.seed(99)
        first_names = [
            "Amanda", "David", "Sara", "Michael", "Ayesha", "Omar", "Liam", "Noah",
            "Emma", "Olivia", "Sophia", "Isabella", "Zara", "Fatima", "Hassan",
            "Imran", "Khalid", "Maya", "Ethan", "Aidan",
        ]
        last_names = [
            "Edwards", "Khan", "Rehman", "Smith", "Johnson", "Williams", "Brown",
            "Patel", "Hussain", "Ali", "Iqbal", "Shah", "Ansari", "Farooq",
            "Ahmed", "Yousaf", "Siddiqui", "Malik", "Chaudhry", "Rauf",
        ]
        fn = np.random.choice(first_names, size=len(df))
        ln = np.random.choice(last_names, size=len(df))
        df["EmployeeName"] = [f"{f} {l}" for f, l in zip(fn, ln)]

    return df


@st.cache_resource
def load_model_and_rag():
    from config import BURNOUT_MODEL_PATH
    if not BURNOUT_MODEL_PATH.exists():
        train_pipeline()

    predictor = BurnoutPredictor.load()
    rag = HRPolicyRAGEngine()
    rag.load_index()
    llm = LLMClient(LLMConfig())
    return predictor, rag, llm


# ---------------- Helper functions ----------------

def compute_risk(df: pd.DataFrame, predictor: BurnoutPredictor) -> pd.DataFrame:
    """Attach BurnoutProb (0–1) and RiskLevel."""
    df_probs = df.copy()
    probs = predictor.predict_proba(df_probs)
    df_probs["BurnoutProb"] = probs
    df_probs["RiskLevel"] = pd.cut(
        df_probs["BurnoutProb"],
        bins=[0.0, 0.4, 0.7, 1.0],
        labels=["Low", "Medium", "High"],
        include_lowest=True,
    )
    return df_probs



def kpi_card(label: str, value: str, sub: str = ""):
    st.markdown(
        f"""
        <div class="kpi-card">
          <div class="kpi-label">{label}</div>
          <div class="kpi-value">{value}</div>
          <div class="kpi-sub">{sub}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def build_employee_label(row: pd.Series) -> str:
    """Label used in employee dropdowns: ID – Name (Role, Dept)."""
    name = row.get("EmployeeName", f"Employee {row['EmployeeID']}")
    job = row.get("JobRole", "Role N/A")
    dept = row.get("Department", "Dept N/A")
    return f"{row['EmployeeID']} – {name} ({job}, {dept})"


# ---------------- Dashboard ----------------

def render_dashboard(df_probs: pd.DataFrame):
    st.markdown("## 🧠 Employee Well-Being Dashboard")
    st.markdown(
        "<span class='section-subtitle'>High-level view of burnout risk across the organization.</span>",
        unsafe_allow_html=True,
    )

    # ---------------- Department filter (for the whole dashboard) ----------------
    all_depts = ["All Departments"] + sorted(df_probs["Department"].dropna().unique().tolist())
    selected_dept = st.selectbox("Filter by Department (Dashboard only)", all_depts, index=0)

    if selected_dept != "All Departments":
        view = df_probs[df_probs["Department"] == selected_dept]
    else:
        view = df_probs

    if view.empty:
        st.warning("No employees in the selected department.")
        return

    # ---------------- KPIs ----------------
    total_employees = len(view)
    risk_counts = (
        view["RiskLevel"]
        .value_counts()
        .reindex(["Low", "Medium", "High"])
        .fillna(0)
        .astype(int)
    )

    low_count = int(risk_counts["Low"])
    med_count = int(risk_counts["Medium"])
    high_count = int(risk_counts["High"])

    risk_rates = risk_counts / total_employees
    avg_sat = 2.57  # demo metric
    turnover_risk_pool = high_count + med_count

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        kpi_card(
            "High Risk Employees",
            f"{high_count}",
            f"{risk_rates['High']*100:.1f}% of selected group",
        )
    with c2:
        kpi_card(
            "Medium Risk",
            f"{med_count}",
            f"{risk_rates['Medium']*100:.1f}% of selected group",
        )
    with c3:
        kpi_card(
            "Estimated Avg. Satisfaction",
            f"{avg_sat:.2f}/4",
            "Demo metric (synthetic)",
        )
    with c4:
        kpi_card(
            "Turnover Risk Pool",
            f"{turnover_risk_pool}",
            "High + Medium risk employees",
        )

    st.markdown("---")

    # ---------------- Risk distribution (pie) + Department Risk (heat bar) ----------------
    col_left, col_right = st.columns([1.1, 1.3])

    # ---- Left: Risk distribution pie ----
    with col_left:
        st.markdown("<div class='section-title'>Risk Distribution</div>", unsafe_allow_html=True)
        risk_df = pd.DataFrame(
            {
                "Risk Level": ["Low", "Medium", "High"],
                "Count": [low_count, med_count, high_count],
            }
        )
        pie = px.pie(
            risk_df,
            names="Risk Level",
            values="Count",
            color="Risk Level",
            color_discrete_map={
                "High": "#ef4444",
                "Medium": "#facc15",
                "Low": "#16a34a",
            },
            hole=0.4,
        )
        pie.update_traces(textinfo="percent+label")
        pie.update_layout(margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(pie, use_container_width=True)

    # ---- Right: Department risk heat-style bar ----
    with col_right:
        st.markdown("<div class='section-title'>Department Risk</div>", unsafe_allow_html=True)

        dept = (
            view
            .groupby("Department")
            .agg(
                HighPct=("RiskLevel", lambda s: (s == "High").mean() * 100.0)
            )
            .reset_index()
            .sort_values("HighPct", ascending=True)
        )

        if dept.empty:
            st.info("No departments available for this filter.")
        else:
            colorscale = [
                [0.0, "#0d7c36"],
                [0.25, "#5ec962"],
                [0.5, "#fee08b"],
                [0.75, "#f46d43"],
                [1.0, "#d73027"],
            ]

            dept["Norm"] = dept["HighPct"] / 100.0

            fig = px.bar(
                dept,
                x="HighPct",
                y="Department",
                orientation="h",
                color="Norm",
                color_continuous_scale=colorscale,
                text=dept["HighPct"].round(1).astype(str) + "%",
            )

            fig.update_traces(
                textposition="outside",
                marker_line_color="rgba(0,0,0,0)",
                marker_line_width=0,
            )

            fig.update_layout(
                xaxis_title="% High-Risk Employees",
                yaxis_title="",
                coloraxis_colorbar=dict(
                    title="Risk %",
                    tickvals=[0, 0.25, 0.5, 0.75, 1.0],
                    ticktext=["0%", "25%", "50%", "75%", "100%"],
                ),
                margin=dict(l=10, r=30, t=30, b=10),
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
            )

            st.plotly_chart(fig, use_container_width=True)
            st.caption("Risk color legend: 🟢 low (0–25%) · 🟡 medium (25–50%) · 🔴 high (>50%).")

    # ---------------- Top 10 High-Risk, High-Performing Employees ----------------

    # Title + small dropdown in same row
    t_col, dd_col = st.columns([4, 2])
    with t_col:
        st.markdown("### 🚨 High-Risk, High-Performing Employees")
    with dd_col:
        table_dept_options = ["All Departments"] + sorted(view["Department"].dropna().unique().tolist())
        selected_table_dept = st.selectbox(
            "",
            table_dept_options,
            index=0,
            label_visibility="collapsed",
        )

    if selected_table_dept == "All Departments":
        hp_view = view
    else:
        hp_view = view[view["Department"] == selected_table_dept]

    if hp_view.empty:
        st.info("No employees found for the selected department and filters.")
        return

    # Decide which column represents performance
    perf_col = None
    for cand in ["PerformanceRating", "PerformanceScore", "PerfRating"]:
        if cand in hp_view.columns:
            perf_col = cand
            break

    if perf_col is not None:
        if perf_col == "PerformanceRating":
            high_perf_mask = hp_view[perf_col] >= 4
        else:
            high_perf_mask = hp_view[perf_col] >= hp_view[perf_col].quantile(0.75)

        subset = hp_view[(hp_view["RiskLevel"] == "High") & high_perf_mask].copy()
    else:
        subset = hp_view[hp_view["RiskLevel"] == "High"].copy()

    if subset.empty:
        st.info("No high-risk, high-performing employees found for the current filters.")
    else:
        subset = subset.sort_values("BurnoutProb", ascending=False).head(10)
        subset["RiskScore"] = (subset["BurnoutProb"] * 100).round(2)

        cols = ["EmployeeID", "EmployeeName", "Department", "JobRole"]
        if perf_col is not None:
            cols.append(perf_col)
        cols.append("RiskScore")
        cols = [c for c in cols if c in subset.columns]

        table = subset[cols].rename(
            columns={perf_col: "Performance"} if perf_col is not None else {}
        )

        # 🔧 drop the original index so we don't show a second "ID-like" column
        table = table.reset_index(drop=True)

        st.dataframe(
            table,
            use_container_width=True,
            height=360,
            hide_index=True, 
        )


# ---------------- Individual Analysis ----------------

def render_individual_view(df_probs: pd.DataFrame):
    """Diagnostic view for a single employee with gauge, risk drivers, and comparison chart."""
    st.markdown("## 🔍 Individual Employee Analysis")
    st.markdown(
        "<span class='section-subtitle'>Deep-dive on a single employee: risk score, key drivers, and context.</span>",
        unsafe_allow_html=True,
    )

    if df_probs.empty:
        st.warning("No employee data available.")
        return

    # --------- FILTER BAR: Department -> Risk Level -> Employee ---------
    cols = st.columns([2, 2, 3])

    # Department filter
    with cols[0]:
        dept_options = ["All Departments"] + sorted(
            df_probs["Department"].dropna().unique().tolist()
        )
        selected_dept = st.selectbox("📁 Department", dept_options, index=0)

    # Apply department filter
    if selected_dept == "All Departments":
        filtered = df_probs.copy()
    else:
        filtered = df_probs[df_probs["Department"] == selected_dept]

    if filtered.empty:
        st.warning("No employees in the selected department.")
        return

    # Risk level filter
    with cols[1]:
        available_levels = set(filtered["RiskLevel"].dropna().unique().tolist())
        base_levels = ["Low", "Medium", "High"]
        level_options = ["All Risk Levels"] + [
            lvl for lvl in base_levels if lvl in available_levels
        ]
        selected_level = st.selectbox("🧱 Risk Level", level_options, index=0)

    if selected_level != "All Risk Levels":
        pool = filtered[filtered["RiskLevel"] == selected_level]
    else:
        pool = filtered

    if pool.empty:
        st.warning("No employees match the selected department and risk level.")
        return

    def _employee_label(row):
        name = row.get("EmployeeName", "Unknown")
        role = row.get("JobRole", "Role N/A")
        dept = row.get("Department", "Dept N/A")
        return f"{int(row['EmployeeID'])} – {name} ({dept}, {role})"

    # Employee selector
    with cols[2]:
        emp_options = pool.apply(_employee_label, axis=1).tolist()
        choice = st.selectbox(
            f"👤 Select Employee ({len(pool)} available)",
            options=emp_options,
            index=0,
        )
        emp_id = int(choice.split(" – ")[0])

    emp = pool[pool["EmployeeID"] == emp_id].iloc[0]

    # --------- CORE METRICS & RISK VALUES ---------
    prob_pct = float(emp["BurnoutProb"] * 100.0)
    risk_level = emp["RiskLevel"]

    wlb_val = emp.get("WorkLifeBalanceScore", np.nan)
    js_val = emp.get("JobSatisfactionScore", np.nan)
    # persona = emp.get("Persona", "N/A")  # Removed: internal data generation label

    # --------- TOP METRICS ROW ---------
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown("**Burnout Risk**")
        st.markdown(f"<h2>{prob_pct:.2f}%</h2>", unsafe_allow_html=True)
        st.caption("Model-predicted burnout probability (0–100%).")

    with col2:
        st.markdown("**Risk Level**")
        emoji = "🟥" if risk_level == "High" else "🟧" if risk_level == "Medium" else "🟩"
        st.markdown(f"<h2>{emoji} {risk_level}</h2>", unsafe_allow_html=True)
        # st.caption(f"Persona: {persona}")  # Removed: internal label

    with col3:
        st.markdown("**Work-Life Balance**")
        if pd.notna(wlb_val):
            if wlb_val <= 1.5:
                emoji = "⚠️"
            elif wlb_val <= 2.5:
                emoji = "😐"
            elif wlb_val <= 3.5:
                emoji = "🙂"
            else:
                emoji = "😄"
            st.markdown(f"<h2>{emoji} {int(wlb_val)}/4</h2>", unsafe_allow_html=True)
        else:
            st.markdown("<h2>–</h2>", unsafe_allow_html=True)
        st.caption("Perceived work-life balance (1–4).")

    with col4:
        st.markdown("**Job Satisfaction**")
        if pd.notna(js_val):
            if js_val <= 1.5:
                emoji = "😟"
            elif js_val <= 2.5:
                emoji = "😐"
            elif js_val <= 3.5:
                emoji = "🙂"
            else:
                emoji = "😄"
            st.markdown(f"<h2>{emoji} {int(js_val)}/4</h2>", unsafe_allow_html=True)
        else:
            st.markdown("<h2>–</h2>", unsafe_allow_html=True)
        st.caption("Overall job satisfaction (1–4).")

    st.markdown("---")

    # --------- BUILD RISK DRIVERS (used in the right column) ---------
    drivers = []

    if "WorkHoursPerWeek" in emp.index and "WorkHoursPerWeek" in df_probs.columns:
        company_hours_avg = df_probs["WorkHoursPerWeek"].mean()
        if emp["WorkHoursPerWeek"] > company_hours_avg + 5:
            drivers.append((
                "Overall workload",
                f"Weekly hours {emp['WorkHoursPerWeek']:.1f}, above company average {company_hours_avg:.1f}."
            ))

    if "StressLevelSelfReport" in emp.index and emp["StressLevelSelfReport"] >= 8:
        drivers.append((
            "High self-reported stress",
            f"Employee reports very high stress: {emp['StressLevelSelfReport']}/10."
        ))

    if "ManagerSupportScore" in emp.index and emp["ManagerSupportScore"] <= 2:
        drivers.append((
            "Low manager support",
            f"Manager support score is low at {emp['ManagerSupportScore']}/5."
        ))

    if "RecognitionFrequency" in emp.index and emp["RecognitionFrequency"] <= 1:
        drivers.append((
            "Low recognition",
            f"Recognition is infrequent: {emp['RecognitionFrequency']} times per month or less."
        ))

    if "SleepHours" in emp.index and emp["SleepHours"] <= 5.5:
        drivers.append((
            "Poor recovery (sleep)",
            f"Average sleep {emp['SleepHours']:.1f} hours per night."
        ))

    if not drivers and risk_level == "High":
        drivers.append((
            "Overall risk pattern",
            "Risk is high even though no single metric is extreme. Address workload, support, and recognition together."
        ))

    # --------- GAUGE (LEFT) + TOP RISK DRIVERS (RIGHT) ---------
    left, right = st.columns([1, 1], gap="large")

    with left:
        st.markdown("### Burnout Risk Score")

        gauge_fig = go.Figure(
            go.Indicator(
                mode="gauge+number",
                value=prob_pct,
                number={"font": {"size": 48}},
                gauge={
                    "axis": {"range": [0, 100]},
                    "bar": {"color": "white"},
                    "steps": [
                        {"range": [0, 40], "color": "green"},
                        {"range": [40, 70], "color": "gold"},
                        {"range": [70, 100], "color": "red"},
                    ],
                    "threshold": {
                        "line": {"color": "white", "width": 4},
                        "thickness": 0.8,
                        "value": prob_pct,
                    },
                },
            )
        )
        gauge_fig.update_layout(
            margin=dict(l=20, r=20, t=30, b=0),
            height=300,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(gauge_fig, use_container_width=True)

    with right:
        st.markdown("### ⚠ Top Risk Drivers")
        st.markdown("<div style='margin-top: 18px;'></div>", unsafe_allow_html=True)

        if drivers:
            for title, desc in drivers:
                st.markdown(f"**• {title}**")
                st.markdown(
                    f"<span style='color:#666; font-size:14px;'>{desc}</span>",
                    unsafe_allow_html=True,
                )
                st.markdown("<br>", unsafe_allow_html=True)
        else:
            st.info("No strong drivers detected. Employee appears balanced.")

    st.markdown("---")
    st.markdown("### 📊 Work Metrics vs Company Average")
    st.markdown("<div style='margin-bottom: 15px;'></div>", unsafe_allow_html=True)

    # --------- BOTTOM: WORK METRICS VS COMPANY AVERAGE (FULL-WIDTH BAR CHART) ---------
    company_df = df_probs  # full company

    def _avg(col):
        return float(company_df[col].mean()) if col in company_df.columns else np.nan

    metrics = [
        ("Hours / Week", "WorkHoursPerWeek"),
        ("Stress (1–10)", "StressLevelSelfReport"),
        ("Manager Support (1–5)", "ManagerSupportScore"),
        ("Recognition (0–5)", "RecognitionFrequency"),
    ]

    rows = []
    for label, colname in metrics:
        emp_val = emp.get(colname, np.nan)
        avg_val = _avg(colname)
        if not np.isnan(emp_val):
            rows.append({"Metric": label, "Type": "Employee", "Value": emp_val})
        if not np.isnan(avg_val):
            rows.append({"Metric": label, "Type": "Company Avg", "Value": avg_val})

    metrics_df = pd.DataFrame(rows).dropna(subset=["Value"])

    if not metrics_df.empty:
        bar_fig = px.bar(
            metrics_df,
            x="Metric",
            y="Value",
            color="Type",
            barmode="group",  # vertical grouped bars
        )
        bar_fig.update_layout(
            margin=dict(l=10, r=10, t=40, b=40),
            height=320,
            legend_title_text="",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(bar_fig, use_container_width=True)
    else:
        st.info("No comparable metrics available for this employee.")


# ---------------- Predictions ----------------

def render_predictions(df_probs: pd.DataFrame):
    st.markdown("## 📊 Predictions Overview")
    st.markdown(
        "<span class='section-subtitle'>Explore predicted burnout risk across employees with flexible filters.</span>",
        unsafe_allow_html=True,
    )

    if df_probs.empty:
        st.warning("No employee data available.")
        return

    # ---------- Filters ----------
    c1, c2, c3, c4 = st.columns(4)

    with c1:
        dept_opts = sorted(df_probs["Department"].dropna().unique().tolist())
        selected_depts = st.multiselect("Departments", dept_opts, default=dept_opts)

    with c2:
        # Persona filter removed - was internal data generation label
        pass

    with c3:
        risk_opts = ["Low", "Medium", "High"]
        selected_risk = st.multiselect("Risk Levels", risk_opts, default=risk_opts)

    with c4:
        min_risk, max_risk = st.slider(
            "Burnout Risk Range (%)",
            0,
            100,
            (40, 99),   # default range
            5,
        )

    view = df_probs.copy()
    view = view[view["Department"].isin(selected_depts)]
    # Persona filter removed
    view = view[view["RiskLevel"].isin(selected_risk)]

    risk_pct = view["BurnoutProb"] * 100
    view = view[(risk_pct >= min_risk) & (risk_pct <= max_risk)]

    if view.empty:
        st.warning("No employees match the selected filters.")
        return

    # ---------- Summary on filtered set ----------
    total_filtered = len(view)
    high_filtered = (view["RiskLevel"] == "High").sum()
    high_pct = (high_filtered / total_filtered) * 100 if total_filtered else 0.0

    st.markdown(
        f"**Summary:** {total_filtered} employees match current filters; "
        f"**{high_filtered}** ({high_pct:.1f}%) are in **High** risk."
    )

    # ---------- Sorting + Top N ----------
    sort_by = st.selectbox(
        "Sort by",
        ["BurnoutProb", "StressLevelSelfReport", "WorkHoursPerWeek"],
        index=0,
    )
    top_n = st.slider("Show top N employees", 10, 500, 100, 10)

    table_view = view.sort_values(sort_by, ascending=False).head(top_n)

    # ---------- Final table: fixed columns ----------
    cols = [
        "EmployeeID",
        "EmployeeName",
        "Department",
        "JobRole",
        "WorkHoursPerWeek",
        "StressLevelSelfReport",
        "ManagerSupportScore",
        "RecognitionFrequency",
        "BurnoutProb",
        "RiskLevel",
    ]

    cols = [c for c in cols if c in table_view.columns]
    out = table_view[cols].copy()

    # Round numeric fields
    if "WorkHoursPerWeek" in out.columns:
        out["WorkHoursPerWeek"] = out["WorkHoursPerWeek"].round(2)
    if "BurnoutProb" in out.columns:
        out["BurnoutProb"] = (out["BurnoutProb"] * 100).round(2)

    # Friendly column names
    rename_map = {
        "EmployeeID": "Employee ID",
        "EmployeeName": "Employee Name",
        "Department": "Department",
        "JobRole": "Job Role",
        "WorkHoursPerWeek": "Weekly Work Hours",
        "StressLevelSelfReport": "Stress Level",
        "ManagerSupportScore": "Management Support Score",
        "RecognitionFrequency": "Recognition Frequency",
        "BurnoutProb": "Burnout Risk (%)",
        "RiskLevel": "Risk Level",
    }
    out = out.rename(columns=rename_map)
    out = out.reset_index(drop=True)

    # ---------- Color RiskLevel column ----------
    # ---------- Formatting + color for Risk Level ----------
    def style_risk(val):
        if val == "High":
            color = "#ef4444"      # red
        elif val == "Medium":
            color = "#facc15"      # amber
        else:
            color = "#22c55e"      # green
        return f"background-color: {color}20; color: {color}; font-weight: 600;"

    # 1) Force numeric formatting to 2 decimals
    numeric_cols = [
        "Weekly Work Hours",
        "Stress Level",
        "Management Support Score",
        "Recognition Frequency",
        "Burnout Risk (%)",
    ]
    fmt_map = {c: "{:.2f}" for c in numeric_cols if c in out.columns}
    styler = out.style.format(fmt_map)

    # 2) Apply risk-level highlighting
    def highlight_risk(col: pd.Series):
        if col.name != "Risk Level":
            return [""] * len(col)
        return [style_risk(v) for v in col]

    styler = styler.apply(highlight_risk, axis=0)

    st.dataframe(
        styler,
        use_container_width=True,
        height=550,
        hide_index=True, 
    )

# ---------------- AI Advisor ----------------
def render_ai_advisor(df_probs: pd.DataFrame, llm: LLMClient):
    st.markdown("## 💬 AI Advisor (Manager Coaching)")
    st.markdown(
        "<span class='section-subtitle'>Workspace for managers: targeted guidance, notes, and AI coaching for a selected employee.</span>",
        unsafe_allow_html=True,
    )

    if df_probs.empty:
        st.warning("No employee data available.")
        return

    # ---------------- FILTERS ----------------
    c1, c2, c3 = st.columns([2, 2, 3])

    with c1:
        dept_opts = ["All Departments"] + sorted(df_probs["Department"].dropna().unique().tolist())
        selected_dept = st.selectbox("Department", dept_opts, index=0)

    with c2:
        risk_opts = ["All Risk Levels", "Low", "Medium", "High"]
        selected_risk = st.selectbox("Risk Level", risk_opts, index=0)

    view = df_probs.copy()
    if selected_dept != "All Departments":
        view = view[view["Department"] == selected_dept]
    if selected_risk != "All Risk Levels":
        view = view[view["RiskLevel"] == selected_risk]

    if view.empty:
        st.warning("No employees match the selected filters.")
        return

    with c3:
        options = view.apply(build_employee_label, axis=1).tolist()
        choice = st.selectbox(
            f"Select Employee ({len(view)} matching filters)",
            options=options,
            index=0,
        )
        emp_id = int(choice.split(" – ")[0])

    emp = view[view["EmployeeID"] == emp_id].iloc[0]
    prob_pct = float(emp["BurnoutProb"] * 100.0)
    risk_level = emp["RiskLevel"]

    # ---------------- HEADER METRICS ROW ----------------
    h1, h2, h3, h4 = st.columns(4)
    with h1:
        st.markdown("**Burnout Risk**")
        st.markdown(f"<h2>{prob_pct:.2f}%</h2>", unsafe_allow_html=True)
        st.caption("Model-predicted burnout risk (0–100%).")

    with h2:
        st.markdown("**Risk Level**")
        emoji = "🟥" if risk_level == "High" else "🟧" if risk_level == "Medium" else "🟩"
        st.markdown(f"<h2>{emoji} {risk_level}</h2>", unsafe_allow_html=True)
        if risk_level == "High":
            st.caption("Critical: requires immediate attention.")
        elif risk_level == "Medium":
            st.caption("Elevated: monitor closely and intervene early.")
        else:
            st.caption("Healthy: maintain and reinforce positive habits.")

    with h3:
        st.markdown("**Department**")
        st.markdown(f"<h3>{emp['Department']}</h3>", unsafe_allow_html=True)
        st.caption(emp.get("JobRole", ""))

    with h4:
        st.markdown("**Tenure**")
        tenure = emp.get("TenureYears", None)
        if tenure is not None:
            st.markdown(f"<h3>{tenure:.1f} years</h3>", unsafe_allow_html=True)
        else:
            st.markdown("<h3>–</h3>", unsafe_allow_html=True)
        st.caption(f"Team size: {emp.get('TeamSize', 'N/A')}")

    st.markdown("---")

    # ---------------- BUILD DRIVERS (for analysis + AI context) ----------------
    # Risk-aware driver detection: adjust sensitivity based on overall risk level
    drivers = []

    # Calculate percentiles for comparison
    if "WorkHoursPerWeek" in df_probs.columns:
        hours_percentile = (df_probs["WorkHoursPerWeek"] < emp["WorkHoursPerWeek"]).mean() * 100
    else:
        hours_percentile = 50
    
    if "StressLevelSelfReport" in df_probs.columns:
        stress_percentile = (df_probs["StressLevelSelfReport"] < emp["StressLevelSelfReport"]).mean() * 100
    else:
        stress_percentile = 50

    # ========== RISK-LEVEL-AWARE THRESHOLDS ==========
    # Adjust sensitivity based on overall risk classification
    
    if risk_level == "High":
        # High risk: Be very sensitive, catch everything
        stress_threshold_critical = 7
        stress_threshold_elevated = 6
        hours_threshold_critical = 48
        hours_threshold_elevated = 45
        hours_percentile_threshold = 75
        support_threshold_critical = 2
        support_threshold_low = 3
        recognition_threshold_critical = 1
        recognition_threshold_low = 2
        sleep_threshold_critical = 5.5
        sleep_threshold_low = 6.5
        wlb_threshold = 2
        satisfaction_threshold = 2
        training_threshold_none = 0
        training_threshold_low = 10
        
    elif risk_level == "Medium":
        # Medium risk: Moderate sensitivity
        stress_threshold_critical = 8
        stress_threshold_elevated = 7
        hours_threshold_critical = 50
        hours_threshold_elevated = 48
        hours_percentile_threshold = 80
        support_threshold_critical = 2
        support_threshold_low = 3
        recognition_threshold_critical = 1
        recognition_threshold_low = 2
        sleep_threshold_critical = 5.5
        sleep_threshold_low = 6.0
        wlb_threshold = 2
        satisfaction_threshold = 2
        training_threshold_none = 0
        training_threshold_low = 5
        
    else:  # Low risk
        # Low risk: Only flag extreme outliers
        stress_threshold_critical = 9
        stress_threshold_elevated = None  # Don't flag elevated for low-risk
        hours_threshold_critical = 55
        hours_threshold_elevated = None
        hours_percentile_threshold = 95  # Only top 5%
        support_threshold_critical = 1
        support_threshold_low = None  # Don't flag "low" support for low-risk
        recognition_threshold_critical = 0  # Only if literally zero
        recognition_threshold_low = None
        sleep_threshold_critical = 4.5  # Only extremely low
        sleep_threshold_low = None
        wlb_threshold = 1  # Only score of 1
        satisfaction_threshold = 1  # Only score of 1
        training_threshold_none = None  # Don't flag for low-risk
        training_threshold_low = None

    # ========== WORKLOAD DETECTION ==========
    if "WorkHoursPerWeek" in df_probs.columns:
        hours_avg = df_probs["WorkHoursPerWeek"].mean()
        
        if emp["WorkHoursPerWeek"] >= hours_threshold_critical or hours_percentile >= hours_percentile_threshold:
            drivers.append(
                ("Excessive Workload (Critical)",
                 f"Weekly hours ({emp['WorkHoursPerWeek']:.1f}) are in the {hours_percentile:.0f}th percentile. Sustained overload is strongly associated with burnout.")
            )
        elif hours_threshold_elevated and (emp["WorkHoursPerWeek"] >= hours_threshold_elevated or hours_percentile >= (hours_percentile_threshold - 10)):
            drivers.append(
                ("Heavy Workload (Significant)",
                 f"Work hours ({emp['WorkHoursPerWeek']:.1f}) are above company average ({hours_avg:.1f}). This can escalate without rebalancing.")
            )

    # ========== STRESS DETECTION ==========
    if "StressLevelSelfReport" in emp.index:
        if emp["StressLevelSelfReport"] >= stress_threshold_critical:
            drivers.append(
                ("High Stress Level (Critical)",
                 f"Self-reported stress ({emp['StressLevelSelfReport']}/10) is very high ({stress_percentile:.0f}th percentile). This is a clear warning sign.")
            )
        elif stress_threshold_elevated and emp["StressLevelSelfReport"] >= stress_threshold_elevated:
            drivers.append(
                ("Elevated Stress",
                 f"Stress level ({emp['StressLevelSelfReport']}/10) is above healthy range. Early support can prevent deterioration.")
            )

    # ========== MANAGER SUPPORT DETECTION ==========
    if "ManagerSupportScore" in emp.index:
        if emp["ManagerSupportScore"] <= support_threshold_critical:
            drivers.append(
                ("Low Manager Support (Critical)",
                 f"Manager support score ({emp['ManagerSupportScore']}/5) is very low. This amplifies burnout from other sources.")
            )
        elif support_threshold_low and emp["ManagerSupportScore"] <= support_threshold_low:
            drivers.append(
                ("Insufficient Manager Support",
                 f"Manager support ({emp['ManagerSupportScore']}/5) is below optimal. Stronger support could buffer stress.")
            )

    # ========== RECOGNITION DETECTION ==========
    if "RecognitionFrequency" in emp.index:
        if emp["RecognitionFrequency"] <= recognition_threshold_critical:
            drivers.append(
                ("Minimal Recognition (Critical)",
                 f"Recognition frequency ({emp['RecognitionFrequency']}/5) is very low. Lack of acknowledgment fuels disengagement.")
            )
        elif recognition_threshold_low and emp["RecognitionFrequency"] <= recognition_threshold_low:
            drivers.append(
                ("Limited Recognition",
                 f"Recognition ({emp['RecognitionFrequency']}/5) is below healthy levels. More acknowledgment could improve engagement.")
            )

    # ========== SLEEP DETECTION ==========
    if "SleepHours" in emp.index:
        if emp["SleepHours"] <= sleep_threshold_critical:
            drivers.append(
                ("Insufficient Sleep (Critical)",
                 f"Reported sleep ({emp['SleepHours']:.1f} hours) is critically low. Poor rest amplifies stress and impairs recovery.")
            )
        elif sleep_threshold_low and emp["SleepHours"] <= sleep_threshold_low:
            drivers.append(
                ("Suboptimal Rest",
                 f"Sleep duration ({emp['SleepHours']:.1f} hours) is below recommended levels. Better rest could improve resilience.")
            )

    # ========== WORK-LIFE BALANCE ==========
    if "WorkLifeBalanceScore" in emp.index and emp["WorkLifeBalanceScore"] <= wlb_threshold:
        drivers.append(
            ("Poor Work-Life Balance",
             f"Work-life balance score ({emp['WorkLifeBalanceScore']}/4) indicates significant strain. Boundaries need strengthening.")
        )

    # ========== JOB SATISFACTION ==========
    if "JobSatisfactionScore" in emp.index and emp["JobSatisfactionScore"] <= satisfaction_threshold:
        drivers.append(
            ("Low Job Satisfaction",
             f"Job satisfaction ({emp['JobSatisfactionScore']}/4) is low. This increases vulnerability to burnout and turnover risk.")
        )

    # ========== DEVELOPMENT / CAREER ==========
    if "TrainingHoursLast6M" in emp.index:
        if training_threshold_none is not None and emp["TrainingHoursLast6M"] == training_threshold_none:
            drivers.append(
                ("Stalled Development (Significant)",
                 "No learning activity in 6 months. Perceived stagnation increases burnout risk.")
            )
        elif training_threshold_low is not None and emp["TrainingHoursLast6M"] <= training_threshold_low:
            drivers.append(
                ("Limited Development",
                 f"Minimal training ({emp['TrainingHoursLast6M']} hours). More development opportunities could improve engagement.")
            )

    # ========== CAREER STAGNATION (Only for High/Medium risk) ==========
    if risk_level in ["High", "Medium"] and "TenureYears" in emp.index and emp["TenureYears"] >= 5:
        if "JobSatisfactionScore" in emp.index and emp["JobSatisfactionScore"] <= 2:
            drivers.append(
                ("Career Stagnation Risk",
                 f"Long tenure ({emp['TenureYears']:.1f} years) combined with low satisfaction suggests possible career stagnation.")
            )

    # ========== FALLBACK FOR HIGH RISK WITH NO DETECTED DRIVERS ==========
    # If model predicts high risk but no specific concerns flagged, surface that
    if risk_level == "High" and prob_pct >= 70 and len(drivers) == 0:
        drivers.append(
            ("Elevated Burnout Risk (Model-Detected)",
             f"ML model predicts {prob_pct:.1f}% burnout risk based on complex feature interactions. "
             "Even if individual metrics appear manageable, the combination suggests vulnerability. "
             "Recommend proactive wellbeing check-in to identify underlying concerns.")
        )
    
    # Medium risk fallback (more conservative)
    if risk_level == "Medium" and prob_pct >= 50 and len(drivers) == 0:
        drivers.append(
            ("Moderate Risk (Model-Detected)",
             f"Model indicates {prob_pct:.1f}% burnout probability. While no single metric is critical, "
             "monitor closely for emerging patterns. Consider scheduling a wellbeing conversation.")
        )

    # ========== "NO CONCERNS" MESSAGE (Only for genuinely low-risk) ==========
    if not drivers:
        if risk_level == "Low" and prob_pct < 30:
            drivers.append(
                ("No Critical Concerns Detected",
                 "Current indicators look healthy. Focus on maintaining balance and early detection of changes.")
            )
        else:
            # Edge case: Medium risk but no drivers
            drivers.append(
                ("Monitoring Recommended",
                 f"No specific concerns flagged, but model indicates {prob_pct:.1f}% risk. "
                 "Continue regular check-ins to catch early warning signs.")
            )

    # ---------------- LAYOUT: AI ANALYSIS + TABS (Actions / Notes / AI Chat) ----------------
    left, right = st.columns([1.3, 1.7])

    # ---- AI Analysis / primary concerns ----
    with left:
        st.markdown("### 📝 AI Analysis")

        # ========== GENERATE AI-POWERED PERSONALIZED SUMMARY (CACHED) ==========
        # Cache summary per employee to avoid regeneration on chat interactions
        summary_cache_key = f"ai_summary_{emp_id}"
        
        if summary_cache_key not in st.session_state:
            try:
                # Build context for personalized summary
                concerns_summary = "; ".join([title for title, _ in drivers[:4]])
                
                summary_prompt = f"""You are an HR advisor analyzing an employee's wellbeing data.

Employee Profile:
- Burnout Risk: {prob_pct:.1f}% ({risk_level} risk level)
- Department: {emp.get('Department', 'N/A')}
- Role: {emp.get('JobRole', 'N/A')}
- Tenure: {emp.get('TenureYears', 'N/A')} years
- Stress Level: {emp.get('StressLevelSelfReport', 'N/A')}/10
- Work Hours: {emp.get('WorkHoursPerWeek', 'N/A')}/week
- Manager Support: {emp.get('ManagerSupportScore', 'N/A')}/5
- Recognition: {emp.get('RecognitionFrequency', 'N/A')}/5
- Job Satisfaction: {emp.get('JobSatisfactionScore', 'N/A')}/4
- Work-Life Balance: {emp.get('WorkLifeBalanceScore', 'N/A')}/4

Top Concerns: {concerns_summary}

Task: Write a 2-3 sentence executive summary of this employee's situation. Be specific to their metrics and situation. Explain what the overall pattern suggests and why it matters. Use bold (**text**) for emphasis on risk level. Keep under 80 words."""

                # Generate personalized summary (only once per employee)
                summary = llm.generate_answer(
                    question=summary_prompt,
                    context_chunks=[]
                )
                
                # Cache it
                st.session_state[summary_cache_key] = summary.strip()
                
            except Exception as e:
                # Fallback to risk-level template if AI fails
                if risk_level == "High":
                    summary = (
                        "This employee is showing **critical burnout risk** based on multiple indicators. "
                        "The pattern suggests sustained pressure with insufficient recovery and a high chance of disengagement or turnover "
                        "if no action is taken."
                    )
                elif risk_level == "Medium":
                    summary = (
                        "This employee is at an **elevated risk of burnout**. Several indicators are trending in the wrong direction, "
                        "but timely intervention can still prevent serious impact."
                    )
                else:
                    summary = (
                        "Current signals indicate a **low burnout risk**. The goal should be to preserve this state and respond quickly "
                        "if early warning signs appear."
                    )
                st.session_state[summary_cache_key] = summary
                st.caption(f"⚠️ Using template (AI unavailable: {str(e)[:50]})")
        
        # Display cached summary
        st.info(st.session_state[summary_cache_key])

        st.markdown("#### 🎯 Primary Concerns")
        for i, (title, desc) in enumerate(drivers, 1):
            with st.expander(title, expanded=(i == 1 and risk_level != "Low")):
                st.write(desc)

    # ---- Tabs: Playbook / Manager Notes / AI Coaching Chat ----
    with right:
        tab1, tab2, tab3 = st.tabs(["📋 Action Playbook", "🗒 Manager Notes", "🤖 AI Coaching Chat"])

        # ----- TAB 1: Action Playbook (Template-Based) -----
        with tab1:
            st.markdown("#### Recommended Actions")
            actions = []

            if risk_level == "High":
                actions.append(
                        ("Immediate Stabilization", [
                            "Schedule an urgent, private 1-on-1 within the next 48 hours.",
                            "Temporarily pause or defer non-critical tasks to reduce overload.",
                            "Align on 1–2 realistic priorities instead of a long task list.",
                            "Encourage use of wellbeing support (EAP, counseling, medical check) where appropriate.",
                        ])
                )
                actions.append(
                        ("Workload & Boundaries", [
                            "Review typical weekly hours and on-call expectations; set clear maximums.",
                            "Protect at least one no-meeting block per week for focus / recovery.",
                            "Avoid weekend work unless truly exceptional, and compensate when it happens.",
                        ])
                )
                actions.append(
                        ("Support & Recognition", [
                            "Ask directly what support would make the biggest difference in the next 2–4 weeks.",
                            "Increase specific, behaviour-based recognition for meaningful contributions.",
                            "Agree on a follow-up check-in date and document the action plan.",
                        ])
                )
            elif risk_level == "Medium":
                actions.append(
                        ("Early Intervention", [
                            "Use the next scheduled 1-on-1 to explicitly discuss workload and stress.",
                            "Clarify priorities and drop low-value work that doesn't move key outcomes.",
                            "Explore whether any personal constraints are adding pressure.",
                        ])
                )
                actions.append(
                        ("Strengthen Support", [
                            "Agree on a simple communication protocol for when work feels overwhelming.",
                            "Offer flexibility where possible (remote days, schedule adjustments).",
                            "Connect the employee to coaching, mentoring, or peer support networks.",
                        ])
                )
                actions.append(
                        ("Growth & Motivation", [
                            "Discuss short-term development goals and one concrete learning opportunity.",
                            "Align work with strengths and interests where possible.",
                            "Recognize recent contributions publicly, not just privately.",
                        ])
                )
            else:
                actions.append(
                        ("Maintain Healthy Conditions", [
                            "Continue regular 1-on-1s with space to talk about workload and wellbeing.",
                            "Protect reasonable working hours and discourage unnecessary overtime.",
                            "Reinforce positive behaviours (breaks, boundaries, use of leave).",
                        ])
                )
                actions.append(
                        ("Support Long-Term Growth", [
                            "Check that the employee sees a clear path for growth over the next 12–18 months.",
                            "Offer stretch tasks matched to their interests without overloading them.",
                        ])
                )
                actions.append(
                        ("Early Warning System", [
                            "Agree on concrete signals to watch for (withdrawal, irritability, missed deadlines).",
                            "Set a periodic wellbeing 'temperature check' (e.g., quick 1–10 stress question).",
                        ])
                )

            for title, bullets in actions:
                with st.expander(title, expanded=(risk_level == "High" and "Immediate" in title)):
                        for b in bullets:
                            st.write(f"- {b}")

        # ----- TAB 2: Manager Notes (session-persistent per employee) -----
        with tab2:
            st.markdown("#### Private Notes for This Employee")
            notes_key = f"advisor_notes_{emp_id}"
            if notes_key not in st.session_state:
                st.session_state[notes_key] = ""

            st.caption("Use this space to track agreements, observations, and follow-ups. Notes stay in this session only.")
            st.session_state[notes_key] = st.text_area(
                "Notes (not shared with the employee or the model)",
                value=st.session_state[notes_key],
                height=180,
                label_visibility="collapsed",
            )

        # ----- TAB 3: AI Coaching Chat (context-aware) -----
        with tab3:
            st.markdown("#### Ask the AI Coach")
            st.caption(
                "Ask for help on how to approach conversations, set boundaries, or design an action plan for this specific employee."
            )

            chat_key = f"advisor_chat_{emp_id}"
            if chat_key not in st.session_state:
                st.session_state[chat_key] = []

            # Show history
            for role, msg in st.session_state[chat_key][-6:]:
                if role == "user":
                    st.markdown(f"**You:** {msg}")
                else:
                    st.markdown(f"**AI Coach:** {msg}")

            user_q = st.text_area(
                "Your question to the AI Coach",
                placeholder="Example: How should I structure a 1:1 conversation with this employee next week?",
                height=100,
                label_visibility="collapsed",
            )

            if st.button("Ask Coach", type="primary"):
                if user_q.strip():
                    # Build a compact context for the LLM
                    context = []
                    context.append(
                        f"Employee risk level: {risk_level}, burnout probability: {prob_pct:.2f}% "
                        f"Department: {emp['Department']}, Role: {emp.get('JobRole','N/A')}"
                    )
                    metric_bits = []
                    if "WorkHoursPerWeek" in emp.index:
                        metric_bits.append(f"Weekly hours: {emp['WorkHoursPerWeek']:.1f}")
                    if "StressLevelSelfReport" in emp.index:
                        metric_bits.append(f"Stress (1–10): {emp['StressLevelSelfReport']}")
                    if "ManagerSupportScore" in emp.index:
                        metric_bits.append(f"Manager support (1–5): {emp['ManagerSupportScore']}")
                    if "RecognitionFrequency" in emp.index:
                        metric_bits.append(f"Recognition freq (0–5): {emp['RecognitionFrequency']}")
                    if metric_bits:
                        context.append("Key metrics: " + ", ".join(metric_bits))
                    if drivers:
                        driver_titles = "; ".join([d[0] for d in drivers[:3]])
                        context.append("Top concerns: " + driver_titles)

                    answer = llm.generate_answer(
                        user_q,
                        context_chunks=context,
                    )

                    st.session_state[chat_key].append(("user", user_q.strip()))
                    st.session_state[chat_key].append(("assistant", answer.strip()))
                    st.rerun()

# ---------------- HR AI Assistant (Policies + Data) ----------------

def render_policy_assistant(rag: HRPolicyRAGEngine, llm: LLMClient, df_probs=None):
    st.markdown("## 🤖 HR AI Assistant")
    st.markdown(
        "<span class='section-subtitle'>Your AI-powered HR assistant. Ask about **company statistics**, **HR policies**, or **wellbeing insights**. Get instant answers grounded in real employee data and policy documents.</span>",
        unsafe_allow_html=True,
    )

    question = st.text_area(
        "Ask anything about HR, wellbeing, or company data",
        placeholder="Examples:\n"
        "• Statistics: 'Which department has the highest burnout risk?'\n"
        "• Policies: 'What's our burnout intervention policy?'\n"
        "• Insights: 'How many employees need immediate attention?'",
        height=120,
    )
    category = st.selectbox(
        "Optional policy category filter",
        ["(Any)", "Mental Health", "Burnout", "Flexible Work", "Performance", "Recognition", "Leave", "Wellbeing"],
        index=0,
    )

    if st.button("Get Answer", type="primary") and question.strip():
        
        # ========== DETECT QUESTION TYPE ==========
        question_lower = question.lower()
        
        # Keywords that indicate data/statistics questions (more flexible)
        data_indicators = [
            # Department queries
            'department', 'dept', 'team', 'division',
            # Comparison/ranking words
            'highest', 'lowest', 'most', 'least', 'best', 'worst', 'top', 'bottom',
            'compare', 'comparison', 'vs', 'versus',
            # Statistics words
            'average', 'mean', 'statistics', 'stats', 'data', 'numbers',
            'how many', 'count', 'total', 'percentage', '%',
            # Distribution words
            'distribution', 'breakdown', 'by ', 'across',
            # Metrics
            'risk', 'burnout', 'stress', 'satisfaction', 'support', 'hours',
            'recognition', 'workload',
            # List/ranking requests
            'show me', 'list', 'rank', 'ranking', 'order'
        ]
        
        # Policy indicators
        policy_indicators = [
            'policy', 'policies', 'procedure', 'procedures', 'guideline', 'guidelines',
            'what should', 'how should', 'what do i', 'what does',
            'process', 'protocol', 'eap', 'leave', 'flexible work',
            'documentation', 'document', 'according to', 'per policy'
        ]
        
        # Count matches for each type
        data_matches = sum(1 for keyword in data_indicators if keyword in question_lower)
        policy_matches = sum(1 for keyword in policy_indicators if keyword in question_lower)
        
        # Determine question type
        # If data matches >= 2, it's likely a data question
        # If policy matches > data matches, it's a policy question
        is_data_question = data_matches >= 2 or (data_matches > 0 and policy_matches == 0)
        
        # Override: if explicitly asking about policy, force policy search
        if any(phrase in question_lower for phrase in ['what is the policy', 'what are the policies', 
                                                         'what is our policy', 'tell me about the policy']):
            is_data_question = False
        
        # ========== HANDLE DATA QUESTIONS ==========
        if is_data_question and df_probs is not None:
            st.markdown("### 📊 Data Analysis")
            st.caption("Analyzing employee burnout data...")
            
            try:
                # Calculate department-level statistics
                dept_stats = df_probs.groupby('Department').agg({
                    'BurnoutProb': ['mean', 'count'],
                    'StressLevelSelfReport': 'mean',
                    'JobSatisfactionScore': 'mean',
                    'WorkHoursPerWeek': 'mean',
                    'ManagerSupportScore': 'mean',
                    'RecognitionFrequency': 'mean'
                }).round(2)
                
                dept_stats.columns = ['Avg_Burnout_Risk', 'Employee_Count', 'Avg_Stress', 
                                     'Avg_Satisfaction', 'Avg_Hours', 'Avg_Support', 'Avg_Recognition']
                dept_stats['Avg_Burnout_Risk'] = (dept_stats['Avg_Burnout_Risk'] * 100).round(1)
                dept_stats['Employee_Count'] = dept_stats['Employee_Count'].astype(int)
                
                # Calculate % High-Risk Employees per department
                high_risk_counts = df_probs[df_probs['RiskLevel'] == 'High'].groupby('Department').size()
                dept_stats['High_Risk_Count'] = high_risk_counts
                dept_stats['High_Risk_Count'] = dept_stats['High_Risk_Count'].fillna(0).astype(int)
                dept_stats['Pct_High_Risk'] = ((dept_stats['High_Risk_Count'] / dept_stats['Employee_Count']) * 100).round(1)
                
                # Sort by average burnout risk (default) or by relevant metric based on question
                question_lower = question.lower()
                
                # Determine if asking for best (lowest risk) or worst (highest risk)
                asking_for_best = any(word in question_lower for word in ['best', 'lowest', 'least', 'good', 'well', 'healthy'])
                asking_for_worst = any(word in question_lower for word in ['worst', 'highest', 'most', 'bad', 'poor', 'problem', 'issue'])
                
                # Default to worst (highest risk) unless explicitly asking for best
                if asking_for_best and not asking_for_worst:
                    sort_ascending = True  # Best = lowest risk
                    sort_desc = 'best performing (lowest risk)'
                else:
                    sort_ascending = False  # Worst = highest risk
                    sort_desc = 'highest risk'
                
                # Determine which metric to sort by
                if 'support' in question_lower:
                    # For support: best = highest support, worst = lowest support
                    dept_stats = dept_stats.sort_values('Avg_Support', ascending=not sort_ascending)
                    sort_metric = f'Manager Support Score ({sort_desc})'
                    
                elif 'stress' in question_lower:
                    # For stress: best = lowest stress, worst = highest stress
                    dept_stats = dept_stats.sort_values('Avg_Stress', ascending=sort_ascending)
                    sort_metric = f'Stress Level ({sort_desc})'
                    
                elif 'satisfaction' in question_lower:
                    # For satisfaction: best = highest satisfaction, worst = lowest satisfaction
                    dept_stats = dept_stats.sort_values('Avg_Satisfaction', ascending=not sort_ascending)
                    sort_metric = f'Job Satisfaction ({sort_desc})'
                    
                elif 'hours' in question_lower or 'workload' in question_lower:
                    # For hours: best = fewer hours, worst = more hours
                    dept_stats = dept_stats.sort_values('Avg_Hours', ascending=sort_ascending)
                    sort_metric = f'Work Hours ({sort_desc})'
                    
                else:
                    # Default: sort by burnout risk
                    dept_stats = dept_stats.sort_values('Avg_Burnout_Risk', ascending=sort_ascending)
                    sort_metric = f'Burnout Risk ({sort_desc})'
                
                # Risk level breakdown by department
                risk_breakdown = df_probs.groupby(['Department', 'RiskLevel']).size().unstack(fill_value=0)
                
                # Build structured data for LLM (easier to parse)
                dept_list = []
                for idx, (dept_name, row) in enumerate(dept_stats.head(10).iterrows(), 1):
                    dept_list.append(f"{idx}. {dept_name}: Avg Risk={row['Avg_Burnout_Risk']}%, Employees={int(row['Employee_Count'])}, "
                                   f"High-Risk={int(row['High_Risk_Count'])} ({row['Pct_High_Risk']}%), "
                                   f"Avg Stress={row['Avg_Stress']:.2f}/10, Avg Support={row['Avg_Support']:.2f}/5")
                
                # Build context for LLM
                context_text = f"""
COMPANY BURNOUT STATISTICS (Sorted by: {sort_metric}):

TOP DEPARTMENTS:
{chr(10).join(dept_list)}

METRICS EXPLANATION:
- Avg Risk: Average burnout probability (0-100%)
- % High-Risk: Percentage of employees classified as High Risk (≥65%)
- Avg Stress: Average stress level (1-10 scale)
- Avg Support: Average manager support score (1-5 scale)

Total Company Stats:
- Total Employees: {len(df_probs)}
- Overall Avg Burnout Risk: {(df_probs['BurnoutProb'].mean() * 100):.1f}%
- Total High-Risk Employees: {len(df_probs[df_probs['RiskLevel'] == 'High'])} ({(len(df_probs[df_probs['RiskLevel'] == 'High']) / len(df_probs) * 100):.1f}%)
"""
                
                # Generate answer using LLM with data context
                data_prompt = f"""You are analyzing employee wellbeing data for an organization. Answer the user's question using ONLY the data provided below.

USER QUESTION: {question}

COMPANY DATA (Sorted by {sort_metric}):
{chr(10).join(dept_list)}

METRICS GUIDE:
- Avg Risk: Average burnout probability (0-100%)
- % High-Risk: Percentage classified as High Risk (≥65%)
- Avg Stress: Average stress level (1-10 scale)
- Avg Support: Average manager support (1-5 scale, higher is better)
- Avg Satisfaction: Average job satisfaction (1-4 scale, higher is better)
- Avg Hours: Average work hours per week

COMPANY TOTALS:
- Total Employees: {len(df_probs)}
- Overall Avg Risk: {(df_probs['BurnoutProb'].mean() * 100):.1f}%
- High-Risk Employees: {len(df_probs[df_probs['RiskLevel'] == 'High'])} ({(len(df_probs[df_probs['RiskLevel'] == 'High']) / len(df_probs) * 100):.1f}%)

INSTRUCTIONS:
1. Use ONLY the department names and exact numbers shown above
2. Answer the specific question asked (don't just list all departments)
3. If asked for "top 3" or "best", list first 3 departments in the sorted order
4. If asked for "worst" or "highest risk", list first 3 departments
5. Explain what the numbers mean in context of the question
6. Be concise (under 150 words)
7. Reference the actual metric being asked about

Answer the question directly and specifically:"""

                answer = llm.generate_answer(
                    question=data_prompt,
                    context_chunks=[]
                )
                
                st.markdown("### Answer")
                st.write(answer)
                
                # Show data visualizations
                with st.expander("📊 View Detailed Statistics"):
                    st.markdown("#### 📋 Department Statistics")
                    st.caption(f"💡 Sorted by: {sort_metric}")
                    st.caption("Two key metrics: (1) **Avg Risk Score** = average across all employees, (2) **% High-Risk** = percentage classified as high-risk")
                    
                    # Rename columns for clarity
                    display_stats = dept_stats.copy()
                    display_stats = display_stats.rename(columns={
                        'Avg_Burnout_Risk': 'Avg Risk Score (%)',
                        'Employee_Count': 'Total Employees',
                        'High_Risk_Count': 'High-Risk Count',
                        'Pct_High_Risk': '% High-Risk',
                        'Avg_Stress': 'Avg Stress (1-10)',
                        'Avg_Satisfaction': 'Avg Satisfaction (1-4)',
                        'Avg_Hours': 'Avg Hours/Week',
                        'Avg_Support': 'Avg Support (1-5)',
                        'Avg_Recognition': 'Avg Recognition (0-5)'
                    })
                    st.dataframe(display_stats, use_container_width=True)
                    
                    st.markdown("#### 📊 Risk Level Distribution by Department")
                    st.dataframe(risk_breakdown, use_container_width=True)
                    
                    # Comparison chart
                    st.markdown("#### 📈 Two Ways to Measure Risk")
                    col1, col2 = st.columns(2)
                    with col1:
                        st.markdown("**Average Burnout Risk Score**")
                        st.caption("Average probability across ALL employees")
                        st.bar_chart(dept_stats['Avg_Burnout_Risk'])
                    with col2:
                        st.markdown("**% High-Risk Employees**")
                        st.caption("Proportion classified as High Risk (≥65%)")
                        st.bar_chart(dept_stats['Pct_High_Risk'])
                
            except Exception as e:
                st.error(f"Error analyzing data: {str(e)}")
                st.info("Falling back to policy search...")
                is_data_question = False  # Fall through to policy search
        
        # ========== HANDLE POLICY QUESTIONS ==========
        if not is_data_question or df_probs is None:
            st.markdown("### 📚 Policy Guidance")
            st.caption("Searching HR policy documents...")
            
            cat_filter = None if category == "(Any)" else category
            chunks = rag.query(question, top_k=5, category_filter=cat_filter)
            context_texts = [c.text for c in chunks]
            answer = llm.generate_answer(question, context_texts)

            st.markdown("### Answer")
            st.write(answer)

            with st.expander("Show retrieved policy context"):
                for i, c in enumerate(chunks, 1):
                    st.markdown(
                        f"**Snippet {i}** – Doc: `{c.doc_id}` | Section: `{c.section}` | Category: `{c.category}`"
                    )
                    st.write(c.text)


# ---------------- Main ----------------

def main():
    predictor, rag, llm = load_model_and_rag()
    df = load_employee_data()
    df_probs = compute_risk(df, predictor)

    # Sidebar navigation only
    with st.sidebar:
        st.markdown("### 🧠 Well-Being System")
        st.markdown("AI-Powered Burnout Prevention")
        st.markdown("---")

        page = st.radio(
            "Navigate to:",
            ["Dashboard", "Individual Analysis", "Predictions", "AI Advisor", "HR AI Assistant"],
            index=0,
        )

        st.markdown("---")   # FIXED
        st.markdown("**Employees monitored**")
        st.markdown(f"<h2 style='margin-top:-0.2rem;'>{len(df_probs):,}</h2>", unsafe_allow_html=True)


    if page == "Dashboard":
        render_dashboard(df_probs)
    elif page == "Individual Analysis":
        render_individual_view(df_probs)
    elif page == "Predictions":
        render_predictions(df_probs)
    elif page == "AI Advisor":
        render_ai_advisor(df_probs, llm)
    elif page == "HR AI Assistant":
        render_policy_assistant(rag, llm, df_probs)


if __name__ == "__main__":
    main()
