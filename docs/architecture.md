# Architecture: TBO Inventory Optimization

**Note:** Simulator outputs under stated assumptions, not TBO real P&L. This is a capital-allocation system (budget = cash at risk); forecasting matters only where it changes a decision.

## End-to-end diagram

`mermaid
flowchart LR
    subgraph Ingestion["Ingestion (simulated / connectors)"]
        H[History events\nsearch/quote/booking/cancel/commit\nexternal signals]
        A[assumptions.yaml\ncontracts, carry, margins, risk]
        C[Candidate cells\nsim/cells.py]
    end

    subgraph FeatureStore["Feature store (point-in-time)"]
        F[build_point_in_time_features\nas_of_ts <= decision, censored only\nno leakage]
    end

    subgraph Engines["Engines"]
        D[Demand engine\nEM unconstraining proxy -> quantiles\nLightGBM per quantile + CQR/scenarios\nforecast/*]
        S[Supply & cost engine\np,c_d,c_b,k,r,a,g,cap,carry\neconomics & contract structures\noptimizer/cell.py, structures.py]
    end

    subgraph Funnel["Decision pipeline"]
        K[Candidate funnel\nscreen delta_m>0, volume/sellout/discount/event\nrank by delta_m/k, k=0 first\noptimizer/planner.py]
        SC[Scenario service\ncorrelated demand scenarios by group\nforecast/scenarios.py]
        O[Optimizer\nfractile F(q*)=(delta_m-carry)/(delta_m+c_o)\nwater-filling + lambda bisection\nmean-CVaR LP / controls\noptimizer/*]
    end

    subgraph Serving["Serving & product"]
        API[Decision API (Go)\nPOST /api/v1/plan, GET /plans/{id}\nPOST /scenarios/run, GET /healthz\napi/]
        UI[UI (Streamlit)\nOpportunity Board, Budget Allocator,\nCell Deep Dive, Scenario Simulator,\nRisk & Execution Monitor\nui/app.py]
    end

    subgraph Exec["Execution hooks & monitor"]
        EX[Execution hooks\nCOMMIT / NEGOTIATE_ALLOTMENT /\nWAIT / RELEASE / DONT_BUY\n+ why-bullets per recommendation]
        M[Monitor\nexposure, pace-vs-plan, release countdown,\nkill-switch, realised-vs-predicted proxy]
    end

    subgraph Eval["Evaluation (eval-only)"]
        EV[eval/*\nbaselines B0-B4, oracle, stress,\nstats/CIs, pricing/timing/exploration]
        GT[(ground_truth.parquet\nlatent demand\nNEVER read by forecast/optimizer)]
    end

    H --> F
    A --> S
    C --> S
    C --> F
    F --> D
    D --> SC
    S --> K
    F --> K
    K --> SC
    SC --> O
    S --> O
    O --> API
    O --> UI
    O --> EX
    EX --> M
    API --> UI
    M --> UI
    SC -. decision scenarios .-> EV
    H -. censored history .-> EV
    GT -. latent realized eval only .-> EV
`

## Component responsibilities

| Stage | Modules / artifacts | Responsibility | Key invariants |
| --- | --- | --- | --- |
| **Ingestion** | sim/*, data/history/*.parquet, config/assumptions.yaml | Generate/load censored event history, external signals, inventory commits; load business assumptions with provenance | observed_sales=min(D,capacity); s_of_ts on rows; data/ground_truth.parquet latent-only |
| **Feature store** | orecast/features.py, orecast/splits.py | Build point-in-time features reconstructible as of decision time; rolling-origin/time splits with embargo = horizon | No leakage (s_of_ts <= decision); no random panel split |
| **Demand engine** | orecast/unconstrain.py, orecast/quantile_model.py, orecast/conformal.py, orecast/scenarios.py, orecast/cancellations.py | Train on censored/unconstrained proxy targets only; produce quantiles/distributions and correlated scenario matrices | Never reads ground truth; correlated group shocks |
| **Supply & cost engine** | optimizer/cell.py, optimizer/demo_phase2.py, optimizer/structures.py, sim/cells.py | Cell economics (delta_m,c_o,carry), contract structures (free/soft/hard/air), capacity/fallback/salvage params | Budget = cash at risk (k=0 free allotment consumes no cash) |
| **Candidate funnel** | optimizer/planner.py:candidate_funnel | Screen/rank feasible, attractive opportunities; prioritize k=0 then delta_m/k | Default: don’t buy unless edge is material; supports WAIT/NEGOTIATE |
| **Scenario service** | orecast/scenarios.py, data/scenarios/*, data/api_demo/scenarios.* | Serve sampled demand scenarios for optimizer/eval/API/UI; pre-baked bundles for reproducibility/demo | Seeded, correlated, CRN for comparisons |
| **Optimizer** | optimizer/fractile.py, optimizer/waterfill.py, optimizer/cvar_lp.py, optimizer/profit.py, optimizer/baselines.py, optimizer/controls.py | Allocate units/cash by critical fractile, budget shadow price lambda (bisection), reserve/constraints, optional mean-CVaR | lambda s.t. sum(k_i*q_i)=Budget when binding; special cases hold |
| **Decision API** | pi/main.go, pi/internal/{alloc,api,data,store} | HTTP plan/scenario/store interface, versioned plans, demo dataset load/export | Go stdlib 
et/http; simulator outputs under stated assumptions |
| **UI** | ui/app.py, scripts/ui.ps1 | Opportunity Board, Budget Allocator, Cell Deep Dive, Scenario Simulator, Risk & Execution Monitor; SIMULATED/LIVE badges and units | Read-only live TBO search fallback; no unstated claims |
| **Execution hooks & monitor** | optimizer/planner.py recommendations, ui/app.py monitor views, 	iming.py | Emit actions/reasons, pace/release/deadline controls and kill-switch logic, realized-vs-predicted proxy | Every recommendation has why-bullets; WAIT/NEGOTIATE/RELEASE supported |
| **Evaluation** | eval/*, pricing.py, 	iming.py, exploration.py, docs/* | Baselines/oracle/stress/stats/pricing/timing/exploration; artifact generation | Ground truth read only inside /eval; seeded, CRN, embargoed splits |

## Data & control flow (summary)
1. **Censored history + assumptions/cells** feed point-in-time features and economics.
2. **Demand engine** fits on censored/unconstrained proxies and emits quantiles/scenarios (correlated by group).
3. **Candidate funnel** filters to attractive, screen-passing cells and ranks by capital efficiency (delta_m/k, k=0 first).
4. **Optimizer** computes unconstrained/budgeted fractiles and water-fills to budget (lambda shadow price), with optional risk/constraints.
5. **Decision API/UI** expose plan, metrics, explanations, and scenario stress views; **execution hooks/monitor** track pace, deadlines, release/kill-switch signals.
6. **Evaluation** compares strategies on held-out/latent scenarios (eval-only) and reports oracle/regret, stress, and CIs.

Assumption: diagram reflects implemented prototype components and interfaces; some production-grade ingestion/connectors and live execution integrations are represented as hooks/simulated views.
