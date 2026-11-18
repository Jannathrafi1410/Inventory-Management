A. WHAT THE SYSTEM DOES
=====================================================================
Given a maximum budget, it decides which travel inventory to pre-buy (hotels mainly, air selectively): whether to commit, which contract structure, how many units, when, at what sell price, and when to release unsold units. It is a CAPITAL-ALLOCATION system, not a forecasting project. Forecasting only matters where it changes a decision.

Business facts (do not invent others):
- TBO is asset-light and holds almost no inventory today. Dynamic-sourcing gross margin is about 5.5% of hotel GTV, so the default answer for most inventory is DON'T BUY.
- Budget means CASH AT RISK, not notional value. A free-release allotment consumes Rs 0.
- Carry cost on locked cash: 12% per year, 120-day default lock.
- Hotels are about 78% of gross profit. Air has near-zero salvage.
- The system must output COMMIT, NEGOTIATE_ALLOTMENT, WAIT, RELEASE and DONT_BUY actions.

=====================================================================
B. CORE MATH (must be implemented exactly; any deviation is a bug)
=====================================================================
Per cell: p = sell price, c_b = pre-buy net cost, c_d = dynamic-sourcing net cost, k = cash prepaid per unit, r = refund fraction if unsold, a = P(dynamic channel covers a shortfall) = 1 - sell-out rate, g = goodwill penalty per unserved unit, D = demand net of cancellations, carry = k * r_annual * lock_days / 365.
  delta_m = (p - c_b) - a*(p - c_d) + (1 - a)*g
  c_o = k * (1 - r)
  Unconstrained optimum: F(q*) = (delta_m - carry) / (delta_m + c_o), clipped to [0,1], q* = min(cap, F^-1(.))
  Budgeted optimum: F_i(q_i*) = (delta_m_i - carry_i - lambda*k_i) / (delta_m_i + c_o_i), clipped, capped. Bisect lambda so sum(k_i*q_i) = Budget. lambda = shadow price of budget. If spend at lambda=0 <= Budget then lambda = 0.
  Expected profit: pi(q) = (p - c_b + k)E[min(D,q)] + [a(p - c_d) - (1-a)g]E[(D-q)+] - k*q + r*k*E[(q-D)+] - carry*q
  Risk: maximize (1-beta)*E[dProfit] + beta*CVaR_alpha(dProfit), via Rockafellar-Uryasev LP (HiGHS). dProfit = profit versus the no-pre-buy baseline.
Special cases that must hold: free release (k=0) gives F=1, take the cap; a=0,g=0 gives (p-c_b)/p; a=1 and c_d=c_b gives q*=0.
Validation anchors:
- DXB-NYE hard buyout: p=21000, c_d=19845, c_b=13495, k=13495, r=0, a=0.55, g=900 -> delta_m=7275, c_o=13495, F=0.350
- DXB-DEC soft release: p=14500, c_d=13702, c_b=9592, k=9592, r=0.70, a=0.78, g=0 -> delta_m=4286, c_o=2878, F=0.598
- Any free allotment -> F=1.0
Contract structures: free_allotment (k=0, r=1, 10-22% discount), soft_release (k=c_b, r=0.7, 20-30%), hard_buyout (k=c_b, r=0, 30-45%), air_bulk_fare (k=c_b, near-zero salvage, 5-20%).

=====================================================================
C. NON-NEGOTIABLE RULES
=====================================================================
1. Demand history is CENSORED: observed = min(D, capacity). Models train only on the censored table. Latent demand lives only in a ground_truth table read exclusively by /eval. No forecast or optimizer module may import it.
2. Every feature is point-in-time correct (as_of_ts <= decision time). No leakage.
3. Time-based splits only, with an embargo gap equal to the forecast horizon.
4. Shocks are correlated by group (destination/region).
5. Everything is seeded and reproducible; use common random numbers when comparing strategies.
6. Every recommendation carries human-readable why-bullets.
7. All numbers are simulator outputs under documented assumptions (config/assumptions.yaml), never TBO's real P&L, and the UI marks them simulated vs live.

=====================================================================
D. EXPECTED REPO AND PHASE SPEC
=====================================================================
Layout: /config (assumptions.yaml, settings), /sim, /data, /forecast, /optimizer, /api (Go), /ui, /eval, /tests, /docs, docker-compose.yml, Makefile, README.md.

P1 Skeleton: typed config loader, seed utility (master seed + label gives independent deterministic streams), Makefile targets (setup, test, sim, plan, ui, demo), docker-compose binding 0.0.0.0, reproducibility test.
P2 Simulator: tables search_event, quote, booking, cancellation, inventory_commit, external_signal (all with as_of_ts); history table (censored only) and separate ground_truth table; 20-40 candidate cells (Dubai NYE/December, Bangkok Songkran, Singapore F1, Maldives, Paris/London/Rome summer, Istanbul, Madrid, Phuket, Bombay-Dubai air bulk, Dubai transfers); bot filtering and look-to-book helpers. Tests: dynamic margin 5.5% +/- 0.3pp, within-group correlation above across-group, history <= capacity, byte-identical regeneration, import-guard test for ground_truth.
P3 Decision core (/optimizer): cell.py, fractile.py, waterfill.py (pre-sorted scenario columns, bisection with doubling upper bound, reserve and concentration cap), profit.py, structures.py. Tests: special cases, anchors, spend <= budget, monotone in lambda, brute-force grid check within 2%, naive newsvendor is worse on DXB-DEC.
P4 Baseline forecast: rolling-origin splits with embargo, seasonal naive, pace model, metrics (WAPE, MASE, pinball, PICP), docs/baseline_metrics.md.
P5 Quantile engine: point-in-time features, EM unconstraining, LightGBM quantiles (0.05-0.95), conformal calibration, beta-binomial cancellations, hierarchical reconciliation, scenario matrix via Gaussian copula with group factors. Targets: beats P4 baseline (or says honestly it does not), 90% PICP within 88-92%, unconstraining closer to latent demand on sold-out dates.
P6 Planner: candidate funnel, plan() producing recommendations (cell_id, action, structure, units, cash, unit_net, sell_price, fractile, delta_m, c_o, demand p10/p50/p90, sellout_prob, why[]), plan-level outputs (lambda, cash_deployed, reserve, structural_uplift, capital_uplift, roi_on_cash, expected cash spoilage, fill-rate uplift), baselines B0-B4, 50-seed evaluation with 90% bootstrap CIs on common random numbers. Gate: beats B1-B3, never exceeds budget, sometimes leaves budget unspent.
P7 Risk: factor-model scenarios (1k-5k, versioned), mean-CVaR LP, monotone efficient frontier, exposure caps, reserve tranche, kill-switch (auto-release if pace < 80% of plan at release-deadline minus 14 days).
P8 Go API: POST /api/v1/plan, GET /api/v1/plans/{id}, POST /api/v1/scenarios/run, GET /healthz; water-filling port; plan store with versioned inputs. Gate: Go output equals Python within 1 unit per cell; budget change re-solves < 1 second on demo set; 50k-cell benchmark reported honestly.
P9 UI: Opportunity Board, Budget Allocator (budget, risk slider, frontier, lambda readout), Cell Deep Dive (fan chart, booking curve, structure comparison), Scenario Simulator, Risk and Execution Monitor, simulated/live badges, optional live TBO API panel with graceful fallback.
P10 Evaluation: stress suite (demand +/-30%, volatility x1.6, correlation x2, sell-out shifts, event cancelled, FX +10%, supplier default, price war -8%, 20% bot contamination), oracle regret, paired bootstrap CIs, 50 seeds, two generator parameterisations, pricing grid and markdown ladder, tranching and priced WAIT, Thompson-sampling exploration (2-5% of budget).
P11 Packaging: README (run in <10 min), docs/architecture.md, docs/assumptions.md, docs/limitations.md, 10-slide deck outline with notes, 3-minute video script, 5-minute demo script, `make demo` smoke test.

=====================================================================