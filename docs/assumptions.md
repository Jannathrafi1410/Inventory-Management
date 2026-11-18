# Assumptions (generated from config/assumptions.yaml)

All values are simulator assumptions (not TBO real P&L). Each entry includes alue, source, and confidence where provided.

**Generation note:** Derived from config/assumptions.yaml in-repo.


## contracts

### contracts.free_allotment
- **contracts.free_allotment.discount_vs_dynamic**: value=[0.1, 0.22] | source="Master context: 10-22% discount band for free-release allotments" | confidence="high"
- **contracts.free_allotment.k**: value=0.0 | source="Master context: free-release allotment consumes Rs 0 cash at risk" | confidence="high"
- **contracts.free_allotment.r**: value=1.0 | source="Master context: free-release allotment refundable in full (option-like)" | confidence="high"
- **contracts.free_allotment.supplier_cap**: value="optional" | source="Master context: optional supplier cap on free allotments" | confidence="medium"

### contracts.soft_release
- **contracts.soft_release.discount_vs_dynamic**: value=[0.2, 0.3] | source="Master context: 20-30% discount band for soft release" | confidence="high"
- **contracts.soft_release.k**: value="c_b" | source="Master context: cash prepaid per unit equals pre-buy net cost" | confidence="high"
- **contracts.soft_release.r**: value=0.7 | source="Master context: 70% refundable if unsold under soft release" | confidence="high"

### contracts.hard_buyout
- **contracts.hard_buyout.discount_vs_dynamic**: value=[0.3, 0.45] | source="Master context: 30-45% discount band for hard buyout" | confidence="high"
- **contracts.hard_buyout.k**: value="c_b" | source="Master context: cash prepaid per unit equals pre-buy net cost" | confidence="high"
- **contracts.hard_buyout.r**: value=0.0 | source="Master context: no refund on unsold hard buyout units" | confidence="high"

### contracts.air_bulk_fare
- **contracts.air_bulk_fare.discount_vs_dynamic**: value=[0.05, 0.2] | source="Master context: 5-20% discount band for air bulk fares" | confidence="high"
- **contracts.air_bulk_fare.k**: value="c_b" | source="Master context: cash prepaid per unit equals pre-buy net cost" | confidence="high"
- **contracts.air_bulk_fare.r**: value=0.0 | source="Master context: near-zero salvage / non-refundable bulk air" | confidence="medium"
- **contracts.air_bulk_fare.salvage_note**: value="near-zero" | source="Master context: air has near-zero salvage" | confidence="high"

## carry
- **carry.annual_rate**: value=0.12 | source="Master context: 12% per year carry cost on locked cash" | confidence="high"
- **carry.lock_days**: value=120 | source="Master context: 120-day lock by default" | confidence="high"

## market
- **market.dynamic_margin**: value=0.055 | source="Master context: dynamic-sourcing gross margin ~5.5% of hotel GTV" | confidence="high"

### market.elasticities
- **market.elasticities.compression**: value=-0.8 | source="Assumption: inelastic compression/peak demand elasticity for hackathon simulator" | confidence="medium"
- **market.elasticities.leisure**: value=-2.5 | source="Assumption: more elastic leisure demand for hackathon simulator" | confidence="medium"

## cancellation

### cancellation.lead_time_dependent_probabilities
- **cancellation.lead_time_dependent_probabilities.description**: value="Cancellation probability decreases with shorter lead time" | source="Assumption: standard travel lead-time cancellation curve for simulator" | confidence="medium"

### cancellation.lead_time_dependent_probabilities.by_lead_time_bucket_days
- **cancellation.lead_time_dependent_probabilities.by_lead_time_bucket_days.gte_60**: value=0.18 | source="Assumed simulator default for long-lead bookings" | confidence="low"
- **cancellation.lead_time_dependent_probabilities.by_lead_time_bucket_days.d30_59**: value=0.12 | source="Assumed simulator default for mid-lead bookings" | confidence="low"
- **cancellation.lead_time_dependent_probabilities.by_lead_time_bucket_days.d7_29**: value=0.07 | source="Assumed simulator default for near-term bookings" | confidence="low"
- **cancellation.lead_time_dependent_probabilities.by_lead_time_bucket_days.lt_7**: value=0.03 | source="Assumed simulator default for last-minute bookings" | confidence="low"

## simulator
- **simulator.number_of_cells**: value=30 | source="Assumption: 20-40 destinations/cells in scope; default 30 for skeleton runs" | confidence="medium"
- **simulator.scenarios**: value=2500 | source="Master context: scenario count for risk/simulation" | confidence="high"
- **simulator.group_sigma**: value=0.2 | source="Master context: shocks correlated by group; assumed 20% group volatility" | confidence="medium"

## risk
- **risk.alpha**: value=0.95 | source="Master context: CVaR confidence level alpha" | confidence="high"
- **risk.beta**: value=0.5 | source="Master context: risk-objective weight on CVaR term" | confidence="high"
