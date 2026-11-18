# 10-Slide Deck Outline (with Speaker Notes)

**Audience:** VoyageHack 4.0 / TBO judges and product/engineering stakeholders  
**Theme:** TBO Inventory Optimization — capital allocation for pre-buy decisions (budget = cash at risk)  
**Disclaimer (use on relevant slides):** All numbers are simulator outputs under stated assumptions, not TBO real P&L.

---

## Slide 1 — Problem and the insight
**Title:** Inventory pre-buy is a capital-allocation problem, not a forecasting contest

**On-slide bullets:**
- TBO is asset-light today; pre-buying inventory commits scarce working capital
- Question: *Given a maximum budget, where should each rupee of cash at risk go?*
- Decisions: commit at all? structure? units? timing? sell price? when to release?
- Default for most inventory: **DON’T BUY** unless the edge is real
- System must also recommend **WAIT / NEGOTIATE_ALLOTMENT / RELEASE**, not only COMMIT

**Visual:** Simple flow: Budget (cash at risk) → candidates → fractile/water-filling → actions + why-bullets

**Speaker notes (≈45–60s):**
- Frame the business pain: committing cash too early/too broadly destroys margin and liquidity; buying nothing leaves margin and fill-rate on the table in specific pockets.
- Insight: forecasting is necessary but not sufficient — the decision is the critical fractile under cash-at-risk, opportunity cost, and fallback supply. The optimizer turns distributions into actions with a shadow price (λ).
- Emphasize capital discipline: free allotments are options (Rs 0 cash), hard buyouts are real bets; recommendations must be explainable.

---

## Slide 2 — TBO economics
**Title:** Business facts that shape every recommendation

**On-slide bullets:**
- Dynamic-sourcing hotel gross margin ~**5.5% of GTV** (baseline)
- Budget = **cash at risk** (not notional); free-release allotment consumes **Rs 0**
- Carry cost on locked cash: **12%/yr**, **120-day** lock default
- Hotels ~**78% of gross profit** (focus); **air has near-zero salvage** (selective)
- Every recommendation includes **why-bullets** and units/currency context

**Visual:** Table comparing contract structures

| Structure | k (cash/unit) |  (refund) | Discount vs dynamic | Role |
| --- | --- | --- | --- | --- |
| free_allotment | 0 | 1.0 | 10–22% | Free option |
| soft_release | c_b | 0.7 | 20–30% | Partial downside protection |
| hard_buyout | c_b | 0.0 | 30–45% | Highest edge, highest risk |
| air_bulk_fare | c_b | ~0 | 5–20% | Selective, near-zero salvage |

**Speaker notes:**
- Connect economics to behavior: when dynamic fallback is reliable/cheap, pre-buy loses; when sell-outs/goodwill or price gaps matter, a controlled commit wins.
- Call out working-capital reality (asset-light, negative working capital context) as why λ (value of next rupee) matters.

---

## Slide 3 — Data and censoring
**Title:** Train on what we observed; evaluate on what was true (simulated)

**On-slide bullets:**
- History is **censored**: observed_sales = min(D, capacity) (sell-outs hide true demand)
- Latent demand exists only in data/ground_truth.parquet — **eval-only**, never read by forecast/optimizer
- Point-in-time features: reconstructible as of s_of_ts (no leakage)
- **Time-based splits** with embargo = forecast horizon (no random panel split)
- Correlated group shocks (destination/region) — diversification is not free

**Visual:** Small chart: latent demand vs capped observed sales on sold-out dates; pipeline box “ground truth locked behind /eval”

**Speaker notes:**
- Explain why censoring breaks naive models and why unconstraining/calibration is needed.
- Stress governance: leakage prevention, embargoed splits, deterministic seeds, and CRN for fair strategy comparisons. All data shown is simulated under stated assumptions.

---

## Slide 4 — Demand engine and calibration
**Title:** From censored signals to decision-ready scenario distributions

**On-slide bullets:**
- Inputs: censored booking/search/quote/inventory/external signals + point-in-time features
- EM/unconstraining proxy for sold-out periods; quantile demand model (LightGBM per quantile)
- Conformal calibration (CQR-style) targeting interval coverage on embargoed uncensored hold-out rows
- Cancellation and lead-time effects incorporated in demand/realization logic
- Output: **correlated scenario matrices** by group feeding the optimizer

**Visual:** Quantile fan chart + PIT/coverage callout (e.g. 90% PICP after calibration, simulator outputs)

**Speaker notes:**
- Position the demand engine as a service to decisions: we care about the right tail percentile and downside, not just mean accuracy.
- Mention honest limitations: coverage depends on split/censoring assumptions; models trained without latent leakage.

---

## Slide 5 — The fractile derivation
**Title:** Buy the right percentile because cash is at risk

**On-slide bullets:**
- Economics per unit: delta_m = (p-c_b) - a*(p-c_d) + (1-a)*g, c_o = k*(1-r), carry = k*r_annual*lock_days/365
- Unconstrained critical fractile: F(q*) = (delta_m - carry) / (delta_m + c_o) (clipped), q* = min(cap, F^{-1}(...))
- Intuition: balance upside of an extra committed unit vs cash destroyed if unsold
- Special cases: free allotment k=0 → F=1; =0,g=0 → (p-c_b)/p; =1,c_d=c_b → q*=0
- Anchors reproduced (simulated): DXB-NYE hard buyout ~0.350, DXB-DEC soft release ~0.598, free allotment 1.0, AIR-BOMDXB ~0.157

**Visual:** Newsvendor-style marginal tradeoff diagram annotated with delta_m, c_o, carry

**Speaker notes:**
- Walk through “sell at the Xth percentile, not the mean” (e.g. DXB NYE) as the explainability hook.
- Reinforce that every COMMIT carries why-bullets derived from these terms.

---

## Slide 6 — Water-filling, lambda and the frontier
**Title:** Allocate scarce cash across cells with a shadow price

**On-slide bullets:**
- Budgeted fractile: F_i = (delta_m_i - carry_i - λ*k_i) / (delta_m_i + c_o_i) (clipped/capped)
- Find λ by bisection so Σ(k_i*q_i)=Budget when binding; λ=0 if non-binding (budget can be unspent)
- λ = expected profit per extra rupee of budget (marginal value of cash)
- Candidate funnel: screen delta_m>0, volume/sellout/discount/event; rank by delta_m/k (k=0 first)
- Mean–CVaR frontier: trade expected uplift vs tail risk (β, α) with reserve/constraints

**Visual:** (1) Cash deployed/reserve/unspent bars + λ callout; (2) efficient frontier (expected uplift vs CVaR95)

**Speaker notes:**
- Demo narrative tie-in: dragging budget changes λ and the allocation; unspent budget is a feature (discipline).
- Note Go API ports water-filling; mean-CVaR LP used in Python risk analysis/controls as documented.

---

## Slide 7 — Risk and the stress table
**Title:** Robustness under adverse regimes (RN vs RA)

**On-slide bullets:**
- Objective: (1-β)*E[dProfit] + β*CVaR_α(dProfit) (Rockafellar–Uryasev); dProfit vs no-pre-buy baseline
- 11 stress cases: demand ±30%, volatility x1.6, correlation x2, sell-out  ±, event cancellation (35%/85%), FX +10% on c_d, supplier default, price war -8%, bot contamination (+20% decision-only)
- Decisions on stressed forecast scenarios; realized on stressed held-out latent scenarios (eval-only)
- Compare **risk-neutral** (water-filling) vs **risk-averse** (mean-CVaR) on uplift, CVaR95, spoilage
- Execution controls: reserves, concentration caps, release/kill-switch logic

**Visual:** Condensed RN vs RA stress table (expected uplift / CVaR95 / spoilage) with 2–3 highlighted rows

**Speaker notes:**
- Explain bot contamination as decision-signal inflation only (truth unchanged) and supplier default as largest-commitment shock.
- Emphasize that risk controls can trade mean for materially better tails/spoilage depending on regime.

---

## Slide 8 — Product screenshots, API and scale funnel
**Title:** From model to operator workflow

**On-slide bullets:**
- UI (Streamlit): Opportunity Board, Budget Allocator, Cell Deep Dive, Scenario Simulator, Risk & Execution Monitor
- Every number labeled with unit/context and SIMULATED/LIVE badge; optional read-only live TBO search with fallback
- Go API (/healthz, /api/v1/plan, /api/v1/plans/{id}, /api/v1/scenarios/run), versioned plan store
- Scale funnel: screened candidates → scenarios → bisection water-filling; pre-sorted scenario matrices
- Demo perf: sub-second demo re-solve; 50k×1k benchmark measured honestly (environment-dependent)

**Visual:** 2–3 annotated UI screenshots + small API request/response snippet

**Speaker notes:**
- Show how explanations (why-bullets, fractiles, p10/p50/p90, sellout prob) surface in UI and API recommendations.
- Be transparent on scale: demo meets interactive target; large synthetic benchmark target not yet met on reference machine.

---

## Slide 9 — Results, oracle regret, CIs and limitations
**Title:** Evidence: uplift with discipline, and what remains uncertain

**On-slide bullets:**
- Baselines B0–B4 vs Planner: never exceed budget, can leave budget unspent, favorable profit vs B1–B3 (simulated, CRN, embargoed held-out eval)
- Oracle (perfect foresight, eval-only): % oracle captured and regret reported (Planner strong capture on simulated setup)
- Paired bootstrap 90% CIs across 2 parameterisations (robustness check); stress and decision artifacts (pricing/timing/exploration)
- **Limitations:** data simulated, parameters assumed, cancellations simplified; no live TBO P&L validation; UI E2E gap

**Visual:** Headline table: Planner vs B0–B4 + oracle % captured/regret; CI robustness callout

**Speaker notes:**
- Report honestly: simulator outputs under stated assumptions. Ground truth used only in /eval.
- Distinguish demo-scale stats run from spec scale-up commands when applicable; point to docs/advanced_eval_results.md.

---

## Slide 10 — 90-day pilot on two corridors with success criteria
**Title:** Path to production value with controlled risk

**On-slide bullets:**
- Scope: 2 corridors (e.g. 1 hotel-heavy leisure/event, 1 mixed/air-selective) with limited cash-at-risk cap
- Integrate: live rates/availability, booking/commit feeds, supplier terms, release deadlines, guardrails
- Operate: daily/weekly re-solve, human-in-the-loop approvals for COMMIT/NEGOTIATE/RELEASE, monitoring + kill-switch
- Calibrate: backtest on realized outcomes, recalibrate demand/cost/elasticity/cancellation, track override reasons
- **Success criteria (pilot):** (1) no budget/constraint breaches; (2) positive incremental contribution vs dynamic-only baseline after carry/spoilage on pilot cells; (3) improved fill-rate where targeted with bounded spoilage/CVaR; (4) forecast/calibration and decision logs auditable; (5) operator adoption (recommendations explainable, actionable)

**Visual:** 90-day timeline (Weeks 1–2 data/connectors, 3–6 shadow mode, 7–10 limited live commits, 11–13 review/scale decision)

**Speaker notes:**
- Close with disciplined rollout: start in shadow mode, cap downside, measure incrementality and tail risk, then expand corridors/structures.
- Reiterate: system’s value is allocating cash at risk with transparent reasons — not claiming guaranteed real-world P&L from simulation alone.
