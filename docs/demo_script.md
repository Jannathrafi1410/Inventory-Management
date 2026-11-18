# 5-Minute Live Demo Script — TBO Inventory Optimization

**Duration:** ~5:00  
**Setup goal:** Have demo data/API/UI ready (or use local artifacts) so each beat is smooth.  
**Global disclaimer to state early and keep visible:** “All numbers shown are simulator outputs under stated assumptions, not TBO real P&L.”

---

## Pre-demo checklist (before going live)
- [ ] Repo opened at C:\...\Inventory-Management; .venv available (or Docker ready).
- [ ] Demo data present: data/history/*.parquet, data/ground_truth.parquet, data/api_demo/* (or run make demo / quick demo beforehand).
- [ ] Optional live UI path: Go API running (cd api; go run . with data/api_demo) and Streamlit UI (.\scripts\ui.ps1 or python -m streamlit run ui/app.py).
- [ ] Have these artifacts/tabs ready to show quickly: docs/baseline_eval.md (or demo output), docs/advanced_eval_results.md, docs/stress_table.md.
- [ ] Fallback if API/UI is unavailable: walk through generated docs/tables and planner CLI output from python -m optimizer.planner --budget 10000000.

**Opening line (5s):**
> “I’ll show a capital-allocation system for TBO: given a budget of cash at risk, it decides what to pre-buy, in what structure, how many units, and when to wait or release. All numbers are simulated under stated assumptions.”

---

## Minute 0:00–0:45 — Frame TBO today (problem + economics)
**Say:**
- “TBO is asset-light with tight working capital; dynamic hotel sourcing margin is ~5.5% of GTV, so the default for most inventory is DON’T BUY.”
- “Budget means **cash at risk**, not notional value. A free-release allotment uses Rs 0 cash. Carry is 12% per year with a 120-day lock by default.”
- “Hotels drive most profit; air has near-zero salvage. Recommendations must include WAIT, NEGOTIATE_ALLOTMENT, and RELEASE — with why-bullets.”

**Show:** README/assumptions snippet or Slide 1–2 points; mention config/assumptions.yaml provenance.

**Key point:** This is an optimizer for scarce cash, not a dashboard of forecasts.

---

## Minute 0:45–1:30 — Drag the budget and watch λ fall
**Show:** **UI → Budget Allocator** (or API plan results at two budgets).
**Do:**
1. Start at a tighter budget (e.g. Rs 50L–1Cr depending on demo data view) and point to **λ (shadow price)**, cash deployed, reserve, unspent.
2. Increase budget slider; re-solve.
3. Highlight that **λ falls** as budget becomes less binding and more cells/units can be funded; if non-binding, **unspent budget remains**.

**Say:**
- “Lambda is the expected profit per extra rupee of budget. When cash is scarce, lambda is high and only the best delta-m per rupee survives. As I add budget, lambda drops — and if spend at lambda zero is within budget, the system leaves money unspent rather than forcing bad buys.”

**Fallback:** run python -m optimizer.planner --budget 5000000 vs --budget 10000000 and compare lambda, cash_deployed, unspent.

---

## Minute 1:30–2:15 — Open Dubai NYE and explain buying the 35th percentile
**Show:** **UI → Opportunity Board / Cell Deep Dive** for DXB-NYE (hard buyout).
**Do:**
1. Open DXB-NYE details: demand fan (p10/p50/p90), committed units q, chosen fractile.
2. Read economics: delta_m, c_o, carry context, sellout probability.
3. Read a why-bullet: “sold at the ~35th percentile, not the mean, because cash is at risk…”

**Say:**
- “For DXB NYE hard buyout, the critical fractile is about 0.35 in our anchor economics — we commit to a lower percentile because unsold prepaid units destroy cash. The system explains the tradeoff: margin advantage over dynamic fallback versus cost of overage.”

**Anchor (simulated):** p=21000, c_d=19845, c_b=13495, k=13495, r=0, a=0.55, g=900 → delta_m=7275, c_o=13495, F≈0.350 (before carry).

---

## Minute 2:15–3:00 — Switch to a free allotment: cash to zero, tail turns positive
**Show:** **UI → Cell Deep Dive contract comparison** (or scenario/contract view) illustrating ree_allotment.
**Do:**
1. Compare hard/soft vs **free_allotment** (k=0, r=1).
2. Point out **F = 1.0**, take available cap (option-like), **cash at risk = Rs 0**.
3. In Scenario Simulator or metrics view, show downside/tail exposure improving when no prepaid cash is committed (relative to cash-funded structures).

**Say:**
- “A free allotment is a free option: no cash prepaid, full refundability, so the optimal fractile becomes 1.0 — take the cap. Cash committed drops to zero and the downside tail improves because we haven’t exposed working capital. That’s why the funnel prioritizes zero-cash structures when they’re attractive.”

**Fallback:** use contract comparison table logic (F=1 when k=0 and edge positive).

---

## Minute 3:00–3:45 — Drag the risk slider; trigger event-cancelled stress
**Show:** **UI → Budget Allocator / Risk controls** and **Scenario Simulator**.
**Do:**
1. Adjust risk control β (mean-CVaR request/context) and/or show frontier intuition (docs/mean_cvar_frontier.csv chart in UI/docs): higher risk aversion targets better tail (CVaR95) often with different mean/spoilage tradeoff.
2. Toggle **“Event cancelled”** stress.
3. Compare before/after: expected uplift, CVaR95, spoilage/cash impact.

**Say:**
- “Risk isn’t just a slider — it changes how we trade expected uplift for tail protection. Under an event-cancelled shock in our simulated assumptions, demand for event segments collapses; the simulator shows immediate P&L, CVaR, and spoilage impact so we can judge robustness and execution controls.”

**Note:** Go allocator currently ports water-filling; UI may display requested β/α and precomputed/simulated risk views as labeled.

---

## Minute 3:45–4:15 — Show a priced WAIT
**Show:** **Opportunity Board / recommendations** with WAIT or NEGOTIATE_ALLOTMENT, and/or docs/timing_policy.md (priced WAIT: buy-now vs after-signal).
**Do:**
1. Open a cell with WAIT/NEGOTIATE_ALLOTMENT.
2. Highlight why-bullets: positive edge but defer/seek better terms; preserve cash flexibility.
3. If showing timing policy artifact, point to “value of waiting” logic (sigma reduction / rate drift assumptions, simulated).

**Say:**
- “The system can recommend WAIT when deferring preserves option value under uncertainty, or NEGOTIATE_ALLOTMENT when economics are positive but structure/cash terms are wrong. That’s priced, explainable patience — not indecision.”

---

## Minute 4:15–5:00 — Close with baselines table and limitations
**Show:** docs/baseline_eval.md (or docs/baseline_eval_demo.md / demo stdout) and docs/advanced_eval_results.md limitations section.
**Do:**
1. Show B0–B4 vs Planner: never exceeds budget, can leave unspent, higher/competitive uplift vs naive baselines, spend/spoilage/ROI/fill-rate columns.
2. Briefly mention oracle gap/regret and stress/CI artifacts (if time): evidence under simulated, embargoed, CRN evaluation.
3. Read limitations clearly.

**Say:**
- “Against baselines, the planner respects cash discipline and improves allocation quality in simulation. We also measure oracle regret, stress behavior, and uncertainty intervals. But these are **simulator outputs under stated assumptions** — **data is simulated, parameters are assumed, and cancellations are simplified**. No live TBO P&L is claimed. Next step is a 90-day pilot on two corridors with real feeds, calibration, guardrails, and audited decisions.”

**Close (5s):**
> “Allocate cash at risk, explain every decision, and know when not to buy.”

---

## Backup Q&A cues
- **Why λ?** Shadow price of budget; ranks marginal rupee efficiency under binding constraints.
- **Why not buy mean demand?** Cash-at-risk newsvendor fractile balances upside vs unrecoverable overage (c_o).
- **Ground truth leakage?** Latent demand is eval-only in /eval; models train on censored history with time splits/embargo.
- **Is this production-ready?** Prototype/demo with simulated evidence; pilot needs live integrations, calibration, and controls.
