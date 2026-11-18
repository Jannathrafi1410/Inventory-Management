# 3-Minute Video Script — TBO Inventory Optimization

**Target length:** ~3:00 (≈450 words spoken, adjust pacing)  
**Tone:** clear, practical, capital-discipline focused  
**On-screen note (persistent/footer when showing numbers):** “Simulator outputs under stated assumptions, not TBO real P&L.”

---

## 0:00–0:25 — Frame: TBO today and the problem
**Visual:** Title card + simple graphic of TBO as B2B distributor (asset-light), hotels driving most gross profit, cash/working-capital icon.
**VO:**
> “TBO is an asset-light B2B travel distributor with negative working capital pressure and a dynamic-sourcing hotel gross margin around five and a half percent of GTV. Pre-buying inventory can create edge — but it also locks cash. This is a capital-allocation problem: given a maximum budget of cash at risk, decide where each rupee should go, and when the right answer is don’t buy, wait, negotiate, or release.”

**On-screen text:** Budget = cash at risk · Default: DON’T BUY · Actions: COMMIT / WAIT / NEGOTIATE_ALLOTMENT / RELEASE

---

## 0:25–0:55 — Budget and λ (shadow price)
**Visual:** Budget Allocator screen. Drag budget slider down/up; highlight cash deployed, reserve, unspent, and λ.
**VO:**
> “In the allocator, I set a budget — cash at risk, not notional value. As I drag the budget, the system re-solves with water-filling and updates lambda — the shadow price, or expected profit per extra rupee. When the budget is tight, lambda rises and only the highest delta-m-per-rupee opportunities survive. When it’s not binding, the system can leave cash unspent. Discipline matters more than spending the full budget.”

**Callouts:** λ (next-rupee shadow value) · cash deployed / reserve / unspent

---

## 0:55–1:25 — DXB NYE: buy the 35th percentile
**Visual:** Cell Deep Dive for DXB-NYE. Show demand fan (p10/p50/p90), chosen fractile ~35%, economics (delta_m, c_o), and why-bullets.
**VO:**
> “Take Dubai New Year’s Eve as a hard buyout. The model doesn’t buy the mean — it buys a demand percentile because cash is at risk. Here the critical fractile comes out around the thirty-fifth percentile: enough to capture upside when fallback is uncertain, but not so much that unsold units destroy cash. The recommendation explains why — margin advantage, cost of overage, sellout probability, and carry.”

**On-screen:** “Sold at the ~35th percentile, not the mean” (simulated)

---

## 1:25–1:50 — Switch to free allotment: cash to zero, tail improves
**Visual:** Contract comparison for same/illustrative cell; toggle/switch view to free allotment (k=0, r=1) showing cash committed = 0 and positive/tail behavior improving in scenario view.
**VO:**
> “Now compare structures. A free allotment has zero cash at risk and full refundability — so the fractile goes to one: take the available cap as an option. Cash committed drops to zero, and the downside tail improves because we’re not exposing prepaid cash. That’s why free options are prioritized when economics justify it.”

**On-screen:** k=0 → F=1.0 · Cash at risk = Rs 0

---

## 1:50–2:20 — Risk slider and event-cancelled stress
**Visual:** Adjust β/risk control (and show frontier context if available); then Scenario Simulator toggle “Event cancelled” with before/after P&L, CVaR95, spoilage.
**VO:**
> “Risk appetite matters. Moving toward a mean-CVaR objective trades some expected uplift for better tail outcomes and lower spoilage in adverse regimes. In the scenario simulator, I trigger an event-cancelled stress — event demand drops sharply in our simulated assumptions — and we instantly see before-and-after P&L, CVaR, and spoilage. Decisions are stress-tested, not just point-estimated.”

**On-screen:** RN vs RA (simulated) · Stress: event cancelled

---

## 2:20–2:45 — Priced WAIT and actionability
**Visual:** Show a WAIT / NEGOTIATE_ALLOTMENT recommendation (or timing/priced-wait view) with why-bullets and timing context.
**VO:**
> “Not every opportunity deserves a commit today. The system can recommend a priced wait — comparing buy-now versus buying after a signal improves — or negotiate softer terms when the edge is positive but cash exposure is unattractive. Every action comes with human-readable reasons.”

**On-screen:** Actions include WAIT / NEGOTIATE_ALLOTMENT / RELEASE

---

## 2:45–3:00 — Results, baselines, and limitations
**Visual:** Baselines/results table (B0–B4 vs Planner, spend discipline, uplift/spoilage/ROI) + limitations callout.
**VO:**
> “On held-out simulated evaluation, the planner respects budget, can leave cash unspent, and outperforms naive baselines while reporting oracle gap and uncertainty. These are simulator outputs under stated assumptions — data is simulated, parameters assumed, and cancellations simplified — but the framework is ready for a disciplined pilot on two corridors.”

**End card:** “TBO Inventory Optimization — allocate cash at risk, explain every decision.” + repo/demo note
