import json
import math
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
API_DEFAULT = os.environ.get("TBO_API_URL", "http://localhost:8080")
DEMO_DIR = ROOT / "data" / "api_demo"
HISTORY_DIR = ROOT / "data" / "history"
FRONTIER_CSV = ROOT / "docs" / "mean_cvar_frontier.csv"
ASSUMPTION_NOTE = "Simulator output under stated assumptions, not TBO real P&L."

st.set_page_config(page_title="TBO Inventory Optimization", layout="wide")


def badge(kind: str = "simulated") -> str:
    return "🟢 LIVE" if kind == "live" else "🟡 SIMULATED"


def fmt_money(x, currency="INR"):
    try:
        v = float(x)
    except Exception:
        return f"{x} {currency}"
    sign = "-" if v < 0 else ""
    v = abs(v)
    if v >= 1e7:
        s = f"{v/1e7:.2f} Cr"
    elif v >= 1e5:
        s = f"{v/1e5:.2f} L"
    else:
        s = f"{v:,.0f}"
    return f"{sign}{currency} {s}"


def fmt_units(x):
    try:
        return f"{float(x):.1f} units"
    except Exception:
        return f"{x} units"


def fmt_pct(x):
    try:
        return f"{float(x):.2f}%"
    except Exception:
        return f"{x}%"


def fmt_pp(x):
    try:
        return f"{float(x):+.2f} pp"
    except Exception:
        return f"{x} pp"


@st.cache_data(show_spinner=False)
def load_demo_cells():
    p = DEMO_DIR / "cells.json"
    if not p.exists():
        return {}, []
    obj = json.loads(p.read_text(encoding="utf-8"))
    cells = obj.get("cells", [])
    return {c["cell_id"]: c for c in cells}, cells


@st.cache_data(show_spinner=False)
def load_demo_scenarios():
    meta_p = DEMO_DIR / "scenarios.meta.json"
    bin_p = DEMO_DIR / "scenarios.bin"
    if not meta_p.exists() or not bin_p.exists():
        return None
    meta = json.loads(meta_p.read_text(encoding="utf-8"))
    s, n = int(meta["s"]), int(meta["n"])
    cols = []
    with open(bin_p, "rb") as f:
        for _ in range(n):
            b = f.read(s * 8)
            if len(b) != s * 8:
                return None
            cols.append(np.frombuffer(b, dtype="<f8").copy())
    mat = np.column_stack(cols) if cols else np.zeros((s, n))
    return {"meta": meta, "demand": mat}


@st.cache_data(show_spinner=False)
def load_frontier():
    if FRONTIER_CSV.exists():
        try:
            df = pd.read_csv(FRONTIER_CSV)
            df["source"] = "docs/mean_cvar_frontier.csv (simulated)"
            return df
        except Exception:
            return None
    return None


@st.cache_data(show_spinner=False)
def load_booking_pace():
    p = HISTORY_DIR / "booking.parquet"
    if not p.exists():
        return None
    try:
        b = pd.read_parquet(p)
        if "status" in b.columns:
            b = b[b["status"] == "confirmed"]
        b = b.copy()
        b["lead_time_days"] = pd.to_numeric(b.get("lead_time_days", 0), errors="coerce").fillna(0)
        grp = b.groupby("lead_time_days").size().reset_index(name="bookings")
        grp = grp.sort_values("lead_time_days", ascending=False)
        grp["cum_booked"] = grp["bookings"].cumsum()
        total = grp["bookings"].sum() or 1
        grp["cum_frac"] = grp["cum_booked"] / total
        return grp
    except Exception:
        return None


def api_get(base, path, timeout=5):
    r = requests.get(base.rstrip("/") + path, timeout=timeout)
    r.raise_for_status()
    return r.json()


def api_post(base, path, payload, timeout=30):
    r = requests.post(base.rstrip("/") + path, json=payload, timeout=timeout)
    r.raise_for_status()
    return r.json()


def ensure_plan(base, req_payload, force=False):
    key = json.dumps(req_payload, sort_keys=True)
    if (not force) and st.session_state.get("plan_key") == key and st.session_state.get("plan"):
        return st.session_state["plan"], None
    try:
        plan = api_post(base, "/api/v1/plan", req_payload, timeout=20)
        st.session_state["plan"] = plan
        st.session_state["plan_key"] = key
        st.session_state["plan_error"] = None
        return plan, None
    except Exception as e:
        st.session_state["plan_error"] = str(e)
        return st.session_state.get("plan"), str(e)


def metric_card(label, value, unit_label, kind="simulated", help_text=None):
    st.caption(f"{badge(kind)} · {unit_label}")
    st.metric(label, value, help=help_text)


def build_plan_request(budget, beta, alpha, segments, reserve_frac, max_dest, max_sup, min_fill):
    return {
        "budget": float(budget),
        "currency": "INR",
        "horizon": {"from": "2026-10-03", "to": "2026-12-31"},
        "risk": {"beta": float(beta), "alpha": float(alpha)},
        "constraints": {
            "max_share_per_destination": float(max_dest),
            "max_share_per_supplier": float(max_sup),
            "min_fill_rate": float(min_fill),
            "allowed_structures": [],
            "reserve_frac": float(reserve_frac),
            "exploration_frac": 0.0,
        },
        "segments": segments if segments else [],
    }


def compute_portfolio_from_scenarios(cells_by_id, recs, demand_mat, cell_ids_order):
    idx = {cid: i for i, cid in enumerate(cell_ids_order)}
    s = demand_mat.shape[0]
    profits = np.zeros(s, dtype=float)
    spoilage = 0.0
    spend = 0.0
    for r in recs:
        cid = r.get("cell_id")
        c = cells_by_id.get(cid)
        if not c or cid not in idx:
            continue
        j = idx[cid]
        d = demand_mat[:, j].astype(float)
        q = float(r.get("units", 0.0) or 0.0)
        p = float(c.get("p", r.get("sell_price", 0.0)) or 0.0)
        cb = float(c.get("c_b", r.get("unit_net", 0.0)) or 0.0)
        k = float(c.get("k", 0.0) or 0.0)
        rr = float(c.get("r", 0.0) or 0.0)
        a = float(c.get("a", 0.0) or 0.0)
        g = float(c.get("g", 0.0) or 0.0)
        cd = float(c.get("c_d", 0.0) or 0.0)
        carry = k * 0.12 * 120.0 / 365.0
        sold = np.minimum(d, q)
        short = np.maximum(d - q, 0.0)
        leftover = np.maximum(q - d, 0.0)
        coeff_sold = p - cb + k
        coeff_short = a * (p - cd) - (1.0 - a) * g
        profits += coeff_sold * sold + coeff_short * short - k * q + rr * k * leftover - carry * q
        spoilage += k * (1.0 - rr) * float(np.mean(leftover)) if s else 0.0
        spend += k * q
    expected = float(np.mean(profits)) if s else 0.0
    if s:
        k_tail = max(1, int(math.ceil((1.0 - 0.95) * s)))
        cvar95 = float(np.mean(np.sort(profits)[:k_tail]))
    else:
        cvar95 = 0.0
    return {"expected_uplift": expected, "cvar95_uplift": cvar95, "spoilage": spoilage, "spend": spend}


def apply_scenario_stress(base_demand, cell_ids, cells_by_id, event_cancelled=False, demand_mult=1.0, fx_mult=1.0, worse_sellouts=False, supplier_default=False):
    d = base_demand.astype(float).copy()
    for j, cid in enumerate(cell_ids):
        c = cells_by_id.get(cid, {})
        mult = demand_mult
        if event_cancelled and str(c.get("segment", "")).lower() == "event":
            mult *= 0.35
        if worse_sellouts:
            mult *= 0.85
        d[:, j] = np.maximum(0.0, d[:, j] * mult)
    econ = {}
    for cid in cell_ids:
        c = dict(cells_by_id.get(cid, {}))
        if fx_mult != 1.0:
            c["c_d"] = float(c.get("c_d", 0.0)) * fx_mult
            c["c_b"] = float(c.get("c_b", 0.0)) * fx_mult
        if supplier_default:
            c["a"] = min(float(c.get("a", 0.0)), 0.2)
            c["r"] = 0.0
        econ[cid] = c
    return d, econ


st.sidebar.title("TBO Inventory Optimization")
st.sidebar.caption(f"{badge('simulated')} UI · {ASSUMPTION_NOTE}")
base_url = st.sidebar.text_input("Go API base URL", value=API_DEFAULT, help="Example: http://localhost:8080")

health = None
try:
    health = api_get(base_url, "/healthz", timeout=2)
    api_status = "connected"
    api_kind = "live"
except Exception:
    api_status = "unreachable (using local demo fallbacks where possible)"
    api_kind = "simulated"

st.sidebar.markdown(f"**API:** {badge(api_kind)} `{api_status}`")
if health:
    st.sidebar.write({k: health.get(k) for k in ["status", "dataset_loaded", "dataset_version"]})

st.sidebar.divider()
st.sidebar.subheader("Allocator controls")
budget = st.sidebar.slider("Budget (INR)", min_value=1_000_000, max_value=30_000_000, value=10_000_000, step=500_000, format="%d")
beta = st.sidebar.slider("Risk aversion β (mean-CVaR, request)", min_value=0.0, max_value=1.0, value=0.0, step=0.05)
alpha = st.sidebar.select_slider("CVaR confidence α", options=[0.90, 0.95, 0.99], value=0.95)
reserve_frac = st.sidebar.slider("Reserve fraction", min_value=0.0, max_value=0.30, value=0.03, step=0.01)
max_dest = st.sidebar.slider("Max share per destination (0=off)", min_value=0.0, max_value=1.0, value=0.0, step=0.05)
max_sup = st.sidebar.slider("Max share per supplier (0=off)", min_value=0.0, max_value=1.0, value=0.0, step=0.05)
min_fill = st.sidebar.slider("Min fill rate advisory (0=off)", min_value=0.0, max_value=1.0, value=0.0, step=0.05)

cells_by_id, cells_list = load_demo_cells()
all_segments = sorted({c.get("segment", "unknown") for c in cells_list}) or ["leisure", "event", "mixed", "business"]
segments = st.sidebar.multiselect("Segments", options=all_segments, default=all_segments)
plan_req = build_plan_request(budget, beta, alpha, segments, reserve_frac, max_dest, max_sup, min_fill)

t0 = time.time()
plan, plan_err = ensure_plan(base_url, plan_req)
solve_ms = (time.time() - t0) * 1000.0
if plan_err:
    st.sidebar.warning(f"Plan API error: {plan_err}")

tabs = st.tabs(["1) Opportunity Board", "2) Budget Allocator", "3) Cell Deep Dive", "4) Scenario Simulator", "5) Risk & Execution Monitor"])
recs = (plan or {}).get("recommendations", [])
rec_df = pd.DataFrame(recs) if recs else pd.DataFrame()
if not rec_df.empty:
    for col in ["units", "cash", "sell_price", "unit_net", "fractile", "delta_m", "c_o", "demand_p10", "demand_p50", "demand_p90", "sellout_prob"]:
        if col in rec_df.columns:
            rec_df[col] = pd.to_numeric(rec_df[col], errors="coerce")

with tabs[0]:
    st.header("Opportunity Board")
    st.caption(f"{badge('simulated')} Ranked opportunities from latest plan · amounts in INR, quantities in units")
    if rec_df.empty:
        st.info("No plan/recommendations available. Start Go API and ensure demo dataset is loaded.")
    else:
        f1, f2, f3, f4 = st.columns(4)
        dests = ["All"] + sorted([d for d in rec_df.get("destination", pd.Series(dtype=str)).dropna().unique().tolist() if d])
        seg_opts = ["All"] + sorted([d for d in rec_df.get("segment", pd.Series(dtype=str)).dropna().unique().tolist() if d])
        actions = ["All"] + sorted([d for d in rec_df.get("action", pd.Series(dtype=str)).dropna().unique().tolist() if d])
        with f1: fd = st.selectbox("Destination", dests, key="ob_dest")
        with f2: fs = st.selectbox("Segment", seg_opts, key="ob_seg")
        with f3: fa = st.selectbox("Action", actions, key="ob_action")
        with f4: search = st.text_input("Search cell", "", key="ob_search")
        view = rec_df.copy()
        if fd != "All" and "destination" in view.columns: view = view[view["destination"] == fd]
        if fs != "All" and "segment" in view.columns: view = view[view["segment"] == fs]
        if fa != "All" and "action" in view.columns: view = view[view["action"] == fa]
        if search and "cell_id" in view.columns: view = view[view["cell_id"].str.contains(search, case=False, na=False)]
        view = view.sort_values(["action", "delta_m", "cash"], ascending=[True, False, False], na_position="last")
        show_cols = [c for c in ["cell_id","destination","segment","action","structure","units","cash","fractile","delta_m","c_o","demand_p50","sellout_prob"] if c in view.columns]
        st.dataframe(view[show_cols], use_container_width=True, hide_index=True)
        st.subheader("Why bullets")
        for _, row in view.head(12).iterrows():
            with st.expander(f"{row.get('cell_id','')} · {row.get('action','')} · {fmt_units(row.get('units',0))} · {fmt_money(row.get('cash',0))}"):
                st.write(f"{badge('simulated')} Economics: delta_m {fmt_money(row.get('delta_m',0))}/unit, c_o {fmt_money(row.get('c_o',0))}/unit, fractile {fmt_pct(100*float(row.get('fractile',0) or 0))}")
                st.write(f"{badge('simulated')} Demand p10/p50/p90: {fmt_units(row.get('demand_p10',0))} / {fmt_units(row.get('demand_p50',0))} / {fmt_units(row.get('demand_p90',0))}, sellout prob {fmt_pct(100*float(row.get('sellout_prob',0) or 0))}")
                for w in (row.get("why") or []): st.markdown(f"- {w}")

with tabs[1]:
    st.header("Budget Allocator")
    st.caption(f"{badge(api_kind)} Plan solved via Go API when available · {ASSUMPTION_NOTE}")
    if plan:
        cash_dep = float(plan.get("cash_deployed", 0.0)); reserve_amt = float(plan.get("reserve", 0.0))
        unspent = max(0.0, float(budget) - cash_dep - reserve_amt); lam = float(plan.get("lambda", 0.0))
        c1,c2,c3,c4 = st.columns(4)
        with c1: metric_card("Cash deployed", fmt_money(cash_dep), "cash, INR")
        with c2: metric_card("Reserve", fmt_money(reserve_amt), "cash, INR")
        with c3: metric_card("Unspent", fmt_money(unspent), "cash, INR")
        with c4: metric_card("Expected uplift", fmt_money(plan.get("expected_uplift", 0.0)), "incremental vs no-pre-buy, INR")
        c5,c6,c7,c8 = st.columns(4)
        with c5: metric_card("CVaR95 uplift", fmt_money(plan.get("cvar95_uplift", 0.0)), "tail metric, INR")
        with c6: metric_card("ROI on cash", fmt_pct(float(plan.get("roi_on_cash", 0.0) or 0.0) * 100.0), "ratio")
        with c7: metric_card("Expected cash spoilage", fmt_money(plan.get("expected_cash_spoilage", 0.0)), "cash, INR")
        with c8: metric_card("Fill-rate uplift", fmt_pp(plan.get("fill_rate_uplift_pp", 0.0)), "percentage points")
        st.markdown(f"**{badge(api_kind)} Lambda:** `{lam:.6g}` — next-rupee shadow value: **{fmt_money(lam)} per marginal INR-equivalent unit** (allocator dual).")
        st.caption(f"Solve metadata: {badge(api_kind)} solve_time_ms={plan.get('solve_time_ms','n/a')} ms · UI round-trip observed ~{solve_ms:.0f} ms · dataset={plan.get('dataset_version','n/a')}")
        if abs(float(beta)) > 1e-9:
            st.info(f"{badge('simulated')} β={beta:.2f} requested; Go allocator uses water-filling port and reports metrics (mean-CVaR LP not ported to Go).")
        if not rec_df.empty:
            top = rec_df.sort_values("cash", ascending=False).head(15)
            fig = px.bar(top, x="cell_id", y="cash", color="action", title="Top allocations by cash deployed (INR)")
            fig.update_layout(yaxis_title="Cash (INR)", xaxis_title="Cell")
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("No plan available. Ensure Go API is running and dataset loaded.")
    st.subheader("Efficient frontier (mean vs CVaR95)")
    frontier = load_frontier()
    if frontier is not None and {"expected_profit", "cvar"}.issubset(frontier.columns):
        figf = go.Figure()
        figf.add_trace(go.Scatter(x=frontier["cvar"], y=frontier["expected_profit"], mode="lines+markers", name="Frontier (β sweep)", text=frontier.get("beta")))
        figf.update_layout(xaxis_title="CVaR95 uplift (INR, simulated)", yaxis_title="Expected uplift (INR, simulated)", title="Mean–CVaR frontier (precomputed, simulated)")
        st.plotly_chart(figf, use_container_width=True)
        st.caption(f"{badge('simulated')} Source: docs/mean_cvar_frontier.csv (precomputed).")
    else:
        st.caption(f"{badge('simulated')} Frontier artifact not found.")

with tabs[2]:
    st.header("Cell Deep Dive")
    st.caption(f"{badge('simulated')} Demand/cost/contract views derived from plan + demo scenarios/history")
    if rec_df.empty:
        st.info("No recommendations loaded.")
    else:
        cell_id = st.selectbox("Select cell", options=rec_df["cell_id"].tolist(), key="dd_cell")
        row = rec_df[rec_df["cell_id"] == cell_id].iloc[0].to_dict()
        cell = cells_by_id.get(cell_id, {}); scen = load_demo_scenarios()
        cA,cB,cC,cD = st.columns(4)
        with cA: metric_card("Action / units", f"{row.get('action')} / {fmt_units(row.get('units',0))}", "decision, units")
        with cB: metric_card("Chosen fractile", fmt_pct(100*float(row.get('fractile',0) or 0)), "demand percentile")
        with cC: metric_card("Cash committed", fmt_money(row.get('cash',0)), "INR")
        with cD: metric_card("Sellout prob @ q", fmt_pct(100*float(row.get('sellout_prob',0) or 0)), "probability")
        st.markdown(f"**{badge('simulated')} Economics:** sell {fmt_money(row.get('sell_price',0))}/unit, net {fmt_money(row.get('unit_net',0))}/unit, delta_m {fmt_money(row.get('delta_m',0))}/unit, c_o {fmt_money(row.get('c_o',0))}/unit, contract `{row.get('structure')}`")
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Demand fan + chosen fractile")
            if scen is not None and cell_id in scen["meta"].get("cell_ids", []):
                j = scen["meta"]["cell_ids"].index(cell_id); d = np.sort(scen["demand"][:, j].astype(float))
                fig = go.Figure(); fig.add_trace(go.Scatter(x=np.arange(len(d)), y=d, mode="lines", name="Demand scenarios (sorted, units)"))
                q = float(row.get("units", 0.0) or 0.0); fig.add_hline(y=q, line_dash="dash", line_color="red", annotation_text=f"Committed q={q:.1f} units")
                for pctl, name in [(10,'p10'),(50,'p50'),(90,'p90')]: fig.add_hline(y=float(np.percentile(d,pctl)), line_dash="dot", annotation_text=f"{name}={np.percentile(d,pctl):.1f} units")
                fig.update_layout(xaxis_title="Scenario rank", yaxis_title="Demand (units)"); st.plotly_chart(fig, use_container_width=True)
            else:
                st.caption(f"{badge('simulated')} Demo scenarios unavailable for this cell.")
        with col2:
            st.subheader("Booking curve vs plan (pace proxy)")
            pace = load_booking_pace(); figb = go.Figure()
            if pace is not None and not pace.empty:
                figb.add_trace(go.Scatter(x=pace["lead_time_days"].values, y=(pace["cum_frac"].values * 100.0), mode="lines+markers", name="Observed booking pace (% of final, simulated history)"))
            q = float(row.get("units", 0.0) or 0.0); p90 = float(row.get('demand_p90',1) or 1)
            figb.add_hline(y=min(100.0, max(0.0, (q / max(p90,1))*100.0)), line_dash="dash", line_color="orange", annotation_text=f"Plan commitment proxy {q:.1f} units")
            figb.update_layout(xaxis_title="Lead time (days)", yaxis_title="Cumulative booked / plan proxy (%)", xaxis=dict(autorange="reversed"))
            st.plotly_chart(figb, use_container_width=True); st.caption(f"{badge('simulated')} Pace from aggregate censored booking history (proxy).")
        col3, col4 = st.columns(2)
        with col3:
            st.subheader("Price-response curve (illustrative)")
            base_p = float(cell.get("p", row.get("sell_price", 0.0)) or 0.0); base_d = float(row.get("demand_p50", 0.0) or 0.0)
            seg = str(cell.get("segment", row.get("segment","leisure"))).lower(); elast = -0.8 if seg == "leisure" else (-2.5 if seg == "business" else -1.4)
            price_grid = np.linspace(0.8*base_p, 1.2*base_p, 21) if base_p>0 else np.array([0.0])
            demand_grid = base_d * np.power(np.maximum(price_grid,1)/max(base_p,1), elast) if base_p>0 else np.array([base_d])
            figp = go.Figure(); figp.add_trace(go.Scatter(x=price_grid, y=demand_grid, mode="lines+markers", name=f"Demand vs price (units), ε={elast} assumed"))
            if base_p>0: figp.add_vline(x=base_p, line_dash="dash", annotation_text=f"Current {fmt_money(base_p)}/unit")
            figp.update_layout(xaxis_title="Sell price (INR/unit)", yaxis_title="Expected demand (units)"); st.plotly_chart(figp, use_container_width=True)
            st.caption(f"{badge('simulated')} Elasticity assumed by segment (illustrative).")
        with col4:
            st.subheader("Contract comparison table")
            p = float(cell.get("p", row.get("sell_price",0)) or 0); cd = float(cell.get("c_d", p*0.945) or 0); cb = float(cell.get("c_b", row.get("unit_net",0)) or 0)
            a = float(cell.get("a", 0.75) or 0.75); g = float(cell.get("g", 0.0) or 0.0)
            cap = float(cell.get("cap", max(float(row.get("units",0) or 0), float(row.get("demand_p90",0) or 0))) or 0)
            rows_c = []
            for name,k,r in [("free_allotment", 0.0, 1.0), ("soft_release", cb, 0.7), ("hard_buyout", cb, 0.0)]:
                delta_m = (p-cb) - a*(p-cd) + (1-a)*g; c_o = k*(1-r); carry = k*r*0.12*120/365; denom = (delta_m + c_o)
                F = 1.0 if k==0 and (delta_m-carry)>0 else (max(0.0, min(1.0, (delta_m-carry)/denom)) if denom>1e-9 else 0.0)
                q_star = min(cap, cap*F) if cap>0 else 0.0
                rows_c.append({"structure": name, "k (INR/unit)": round(k,2), "r (salvage frac)": r, "delta_m (INR/unit)": round(delta_m,2), "c_o (INR/unit)": round(c_o,2), "fractile F": round(F,4), "q* cap-aware (units)": round(q_star,2), "cash@q* (INR)": round(k*q_star,2)})
            st.dataframe(pd.DataFrame(rows_c), use_container_width=True, hide_index=True)
            st.caption(f"{badge('simulated')} Contract comparison (lock 120 days, carry 12%/yr).")
        st.subheader("Why bullets")
        for w in (row.get("why") or []): st.markdown(f"- {w}")

with tabs[3]:
    st.header("Scenario Simulator")
    st.caption(f"{badge('simulated')} P&L before/after using pre-baked demo scenario set with instant stress transforms")
    scen = load_demo_scenarios()
    if scen is None or rec_df.empty:
        st.info("Demo scenarios or plan unavailable. Ensure data/api_demo exists and a plan is loaded.")
    else:
        tcols = st.columns(5)
        with tcols[0]: ev_cancel = st.checkbox("Event cancelled", value=False)
        with tcols[1]: dem_down = st.checkbox("Demand -30%", value=False)
        with tcols[2]: fx_up = st.checkbox("FX +10%", value=False)
        with tcols[3]: worse = st.checkbox("Worse sell-outs", value=False)
        with tcols[4]: sup_def = st.checkbox("Supplier default", value=False)
        demand = scen["demand"]; ids = scen["meta"].get("cell_ids", [])
        d_stress, econ_cells = apply_scenario_stress(demand, ids, cells_by_id, event_cancelled=ev_cancel, demand_mult=(0.7 if dem_down else 1.0), fx_mult=(1.1 if fx_up else 1.0), worse_sellouts=worse, supplier_default=sup_def)
        base_metrics = compute_portfolio_from_scenarios(cells_by_id, recs, demand, ids)
        after = compute_portfolio_from_scenarios(econ_cells, recs, d_stress, ids)
        no_prebuy_recs = []
        for r in recs:
            rr = dict(r); c = cells_by_id.get(r.get("cell_id"), {})
            rr["units"] = 0.0 if float(c.get("k", 0.0)) > 0 else float(r.get("units", 0.0) or 0.0)
            no_prebuy_recs.append(rr)
        before = compute_portfolio_from_scenarios(econ_cells, no_prebuy_recs, d_stress, ids)
        b1,b2,b3,b4 = st.columns(4)
        with b1: metric_card("Before (no pre-buy)", fmt_money(before["expected_uplift"]), "expected incremental, INR")
        with b2: metric_card("After (plan)", fmt_money(after["expected_uplift"]), "expected incremental, INR")
        with b3: metric_card("Delta P&L", fmt_money(after["expected_uplift"]-before["expected_uplift"]), "after-before, INR")
        with b4: metric_card("CVaR95 after", fmt_money(after["cvar95_uplift"]), "tail, INR")
        s1,s2,s3 = st.columns(3)
        with s1: metric_card("Base expected uplift", fmt_money(base_metrics["expected_uplift"]), "plan on base scenarios, INR")
        with s2: metric_card("Stress spoilage", fmt_money(after["spoilage"]), "expected cash spoilage, INR")
        with s3: metric_card("Stress spend", fmt_money(after["spend"]), "cash deployed, INR")
        comp = pd.DataFrame([
            {"case": "No pre-buy (stressed)", "expected_uplift_INR": before["expected_uplift"], "cvar95_INR": before["cvar95_uplift"], "spoilage_INR": before["spoilage"]},
            {"case": "Plan (stressed)", "expected_uplift_INR": after["expected_uplift"], "cvar95_INR": after["cvar95_uplift"], "spoilage_INR": after["spoilage"]},
            {"case": "Plan (base)", "expected_uplift_INR": base_metrics["expected_uplift"], "cvar95_INR": base_metrics["cvar95_uplift"], "spoilage_INR": base_metrics["spoilage"]},
        ])
        st.dataframe(comp, use_container_width=True, hide_index=True)
        st.caption(f"{badge('simulated')} Event cancelled→event demand×0.35, demand -30%, worse sell-outs→demand×0.85 proxy, FX +10%→costs×1.10, supplier default→availability/salvage shocked.")

with tabs[4]:
    st.header("Risk and Execution Monitor")
    st.caption(f"{badge('simulated')} Exposure/pace/release views derived from plan and simulated history")
    if rec_df.empty:
        st.info("No plan loaded.")
    else:
        ex1, ex2 = st.columns(2)
        with ex1:
            if "destination" in rec_df.columns:
                by_dest = rec_df.groupby("destination", dropna=False)["cash"].sum().reset_index()
                figd = px.bar(by_dest, x="destination", y="cash", title="Exposure by destination (cash, INR)")
                figd.update_layout(yaxis_title="Cash (INR)"); st.plotly_chart(figd, use_container_width=True)
        with ex2:
            sup_col = "group" if "group" in rec_df.columns else ("supplier_group" if "supplier_group" in rec_df.columns else None)
            if sup_col and sup_col in rec_df.columns:
                by_sup = rec_df.groupby(sup_col, dropna=False)["cash"].sum().reset_index()
                figs = px.bar(by_sup, x=sup_col, y="cash", title="Exposure by supplier/group (cash, INR)")
                figs.update_layout(yaxis_title="Cash (INR)", xaxis_title=sup_col); st.plotly_chart(figs, use_container_width=True)
            else:
                st.caption(f"{badge('simulated')} Supplier/group exposure not available in response.")
        st.subheader("Pace-vs-plan heatmap (simulated)")
        hm = rec_df.copy(); hm["plan_units"] = pd.to_numeric(hm.get("units", 0), errors="coerce").fillna(0)
        d50 = pd.to_numeric(hm.get("demand_p50", 0), errors="coerce").replace(0, np.nan)
        hm["pace_index_pct"] = ((hm["plan_units"] / d50) * 100.0).fillna(0)
        idx_col = "destination" if "destination" in hm.columns else "cell_id"
        pivot = hm.pivot_table(index=idx_col, columns="cell_id", values="pace_index_pct", aggfunc="mean").fillna(0)
        fig_h = px.imshow(pivot, aspect="auto", color_continuous_scale="RdYlGn", title="Commitment vs demand p50 pace index (%)")
        st.plotly_chart(fig_h, use_container_width=True)
        st.caption(f"{badge('simulated')} Pace index = committed units / demand p50 (%), plan proxy.")
        st.subheader("Release-deadline countdown and actions")
        release_deadline_days = st.slider("Days to release deadline (simulated)", 0, 30, 10)
        lead_days = 14; threshold = 0.80
        pace_actual = st.slider("Observed pace vs plan to date (%) (simulated)", 0, 150, 75)
        trigger = (release_deadline_days <= lead_days) and (pace_actual/100.0 < threshold)
        m1,m2,m3 = st.columns(3)
        with m1: metric_card("Countdown to deadline", f"{release_deadline_days} days", "time")
        with m2: metric_card("Pace vs plan", fmt_pct(pace_actual), "observed/plan to date")
        with m3: metric_card("Kill-switch", "TRIGGERED" if trigger else "armed/monitoring", "control state")
        if trigger:
            st.error(f"{badge('simulated')} Kill-switch: pace < {int(threshold*100)}% of plan within {lead_days} days of release deadline. Action: RELEASE or renegotiate exposed low-pace cash commitments.")
        else:
            st.success(f"{badge('simulated')} No auto-release trigger now. Monitor and prepare RELEASE/NEGOTIATE_ALLOTMENT if pace falls below {int(threshold*100)}%.")
        st.subheader("Realised vs predicted P&L (simulated proxy)")
        scen = load_demo_scenarios()
        if scen is not None:
            ids = scen["meta"].get("cell_ids", []); d = scen["demand"]
            pred = compute_portfolio_from_scenarios(cells_by_id, recs, d, ids)
            cut = max(1, int(0.2*d.shape[0])); realised = compute_portfolio_from_scenarios(cells_by_id, recs, d[:cut,:], ids)
            rv = pd.DataFrame([
                {"metric":"Expected uplift (INR)", "predicted": pred["expected_uplift"], "realised_proxy": realised["expected_uplift"]},
                {"metric":"CVaR95 uplift (INR)", "predicted": pred["cvar95_uplift"], "realised_proxy": realised["cvar95_uplift"]},
                {"metric":"Cash spoilage (INR)", "predicted": pred["spoilage"], "realised_proxy": realised["spoilage"]},
            ])
            rv["variance"] = rv["realised_proxy"] - rv["predicted"]
            st.dataframe(rv, use_container_width=True, hide_index=True)
            st.caption(f"{badge('simulated')} Realised values are proxy samples from pre-baked demo scenarios (not live actuals).")
        else:
            st.caption(f"{badge('simulated')} Demo scenarios unavailable.")

st.divider()
st.header("Optional: Live TBO hotel search panel (read-only)")
st.caption(f"{badge('live')} Attempted when credentials/config provided; graceful fallback to simulated/demo")
with st.expander("Live TBO hotel search (read-only)", expanded=False):
    st.write(f"{badge('simulated')} Read-only panel. No credentials stored.")
    tbo_url = st.text_input("TBO search endpoint URL", value=os.environ.get("TBO_SEARCH_URL", ""))
    tbo_token = st.text_input("Bearer token (optional)", value=os.environ.get("TBO_TOKEN", ""), type="password")
    city = st.text_input("City/destination", value="Dubai")
    checkin = st.text_input("Check-in (YYYY-MM-DD)", value="2026-12-20")
    checkout = st.text_input("Check-out (YYYY-MM-DD)", value="2026-12-22")
    if st.button("Search live TBO inventory", key="tbo_search"):
        if not tbo_url:
            st.warning(f"{badge('simulated')} No live endpoint configured. Showing fallback demo results.")
            demo = pd.DataFrame([
                {"hotel":"Demo Dubai Marina", "city":city, "checkin":checkin, "checkout":checkout, "available_rooms":12, "net_rate_INR":12500, "source":"fallback-demo"},
                {"hotel":"Demo Palm Resort", "city":city, "checkin":checkin, "checkout":checkout, "available_rooms":5, "net_rate_INR":18900, "source":"fallback-demo"},
            ])
            st.dataframe(demo, use_container_width=True, hide_index=True)
        else:
            try:
                headers = {"Authorization": f"Bearer {tbo_token}"} if tbo_token else {}
                resp = requests.get(tbo_url, params={"city": city, "checkin": checkin, "checkout": checkout}, headers=headers, timeout=8)
                resp.raise_for_status()
                data = resp.json()
                st.success(f"{badge('live')} Live response received.")
                st.json(data)
            except Exception as e:
                st.warning(f"{badge('simulated')} Live search failed gracefully: {e}. Showing fallback demo results.")
                demo = pd.DataFrame([
                    {"hotel":"Demo Dubai Marina", "city":city, "checkin":checkin, "checkout":checkout, "available_rooms":12, "net_rate_INR":12500, "source":"fallback-demo"},
                    {"hotel":"Demo Palm Resort", "city":city, "checkin":checkin, "checkout":checkout, "available_rooms":5, "net_rate_INR":18900, "source":"fallback-demo"},
                ])
                st.dataframe(demo, use_container_width=True, hide_index=True)

st.caption(f"{badge('simulated')} All UI values are simulator outputs under stated assumptions unless marked LIVE. No number shown without a unit/context.")
