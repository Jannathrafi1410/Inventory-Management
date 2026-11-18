package alloc

import (
	"math"
	"sort"
)

// PortfolioMetrics holds aggregate plan metrics computed on scenario matrix.
// All values are incremental (dProfit) vs no-pre-buy baseline; simulator outputs under stated assumptions.
type PortfolioMetrics struct {
	ExpectedUplift      float64   `json:"expected_uplift"`
	CVaR95Uplift        float64   `json:"cvar95_uplift"`
	ExpectedCashSpoilage float64  `json:"expected_cash_spoilage"`
	FillRateUpliftPP    float64   `json:"fill_rate_uplift_pp"`
	StructuralUplift    float64   `json:"structural_uplift"`
	CapitalUplift       float64   `json:"capital_uplift"`
	ROIonCash           float64   `json:"roi_on_cash"`
	PerScenarioProfit   []float64 `json:"-"`
}

// PerScenarioProfit returns per-scenario dProfit for a cell at quantity q.
// pi(q) = (p-c_b+k)*min(D,q) + [a*(p-c_d)-(1-a)*g]*(D-q)+ - k*q + r*k*(q-D)+ - carry*q
func PerScenarioProfit(c Cell, q float64, demand []float64, lockDays int, rAnnual float64) []float64 {
	qf := math.Max(q, 0)
	carry := c.Carry(lockDays, rAnnual)
	coeffSold := c.P - c.CB + c.K
	coeffShort := c.A*(c.P-c.CD) - (1.0-c.A)*c.G
	out := make([]float64, len(demand))
	for i, d := range demand {
		sold := math.Min(d, qf)
		short := math.Max(d-qf, 0)
		leftover := math.Max(qf-d, 0)
		out[i] = coeffSold*sold + coeffShort*short - c.K*qf + c.R*c.K*leftover - carry*qf
	}
	return out
}

// PortfolioPerScenarioProfit sums per-cell per-scenario dProfit.
func PortfolioPerScenarioProfit(cells []Cell, quantities []float64, sortedDemand [][]float64, lockDays int, rAnnual float64) []float64 {
	if len(cells) == 0 || len(sortedDemand) == 0 {
		return nil
	}
	s := len(sortedDemand[0])
	total := make([]float64, s)
	for i, c := range cells {
		if i >= len(sortedDemand) || i >= len(quantities) {
			continue
		}
		ps := PerScenarioProfit(c, quantities[i], sortedDemand[i], lockDays, rAnnual)
		for j := 0; j < s && j < len(ps); j++ {
			total[j] += ps[j]
		}
	}
	return total
}

func mean(xs []float64) float64 {
	if len(xs) == 0 {
		return 0
	}
	sum := 0.0
	for _, x := range xs {
		sum += x
	}
	return sum / float64(len(xs))
}

// BruteForceCVaR computes sample CVaR_alpha as mean of worst (1-alpha) tail (lower tail of profits).
func BruteForceCVaR(profits []float64, alpha float64) float64 {
	if len(profits) == 0 {
		return 0
	}
	if alpha <= 0 {
		alpha = 1e-6
	}
	if alpha >= 1 {
		alpha = 1 - 1e-6
	}
	p := append([]float64(nil), profits...)
	sort.Float64s(p)
	s := len(p)
	k := int(math.Ceil((1.0 - alpha) * float64(s)))
	if k < 1 {
		k = 1
	}
	if k > s {
		k = s
	}
	sum := 0.0
	for i := 0; i < k; i++ {
		sum += p[i]
	}
	return sum / float64(k)
}

// EvaluatePortfolio computes plan metrics including structural/capital uplift decomposition.
// structural_uplift = E[dProfit] of zero-cash cells only (free options/allotments).
// capital_uplift = E[dProfit] of cash-consuming cells only.
// expected_uplift = total incremental profit mean; fill_rate_uplift_pp in percentage points.
func EvaluatePortfolio(cells []Cell, quantities []float64, sortedDemand [][]float64, lockDays int, rAnnual float64, alpha float64) PortfolioMetrics {
	n := len(cells)
	if n == 0 {
		return PortfolioMetrics{}
	}
	totalScen := PortfolioPerScenarioProfit(cells, quantities, sortedDemand, lockDays, rAnnual)
	expected := mean(totalScen)
	cvar := BruteForceCVaR(totalScen, alpha)

	// spoilage and fill-rate uplift
	spoilage := 0.0
	baselineFill := 0.0
	withFill := 0.0
	s := 0
	if len(sortedDemand) > 0 {
		s = len(sortedDemand[0])
	}
	for i, c := range cells {
		qi := 0.0
		if i < len(quantities) {
			qi = quantities[i]
		}
		col := []float64{}
		if i < len(sortedDemand) {
			col = sortedDemand[i]
		}
		if len(col) == 0 || s == 0 {
			continue
		}
		meanLeftover := 0.0
		bf := 0.0
		wf := 0.0
		for _, d := range col {
			leftover := math.Max(qi-d, 0)
			meanLeftover += leftover
			servedNo := c.A * d
			servedYes := math.Min(d, qi) + c.A*math.Max(d-qi, 0)
			denom := math.Max(d, 1e-9)
			// clip to [0,1]
			bv := math.Max(0, math.Min(1, servedNo/denom))
			wv := math.Max(0, math.Min(1, servedYes/denom))
			bf += bv
			wf += wv
		}
		inv := 1.0 / float64(len(col))
		meanLeftover *= inv
		spoilage += c.K * (1.0 - c.R) * meanLeftover
		baselineFill += bf * inv
		withFill += wf * inv
	}
	fillUplift := (withFill - baselineFill) / float64(n) * 100.0 // percentage points

	zeroQ := append([]float64(nil), quantities...)
	cashQ := append([]float64(nil), quantities...)
	for i, c := range cells {
		if c.K > 0 {
			if i < len(zeroQ) {
				zeroQ[i] = 0
			}
		} else {
			if i < len(cashQ) {
				cashQ[i] = 0
			}
		}
	}
	structural := mean(PortfolioPerScenarioProfit(cells, zeroQ, sortedDemand, lockDays, rAnnual))
	capital := mean(PortfolioPerScenarioProfit(cells, cashQ, sortedDemand, lockDays, rAnnual))
	cashDeployed := 0.0
	for i, c := range cells {
		if i < len(quantities) {
			cashDeployed += c.K * quantities[i]
		}
	}
	roi := 0.0
	if cashDeployed > 1e-9 {
		roi = capital / cashDeployed
	}
	return PortfolioMetrics{
		ExpectedUplift:       expected,
		CVaR95Uplift:         cvar,
		ExpectedCashSpoilage: spoilage,
		FillRateUpliftPP:     fillUplift,
		StructuralUplift:     structural,
		CapitalUplift:        capital,
		ROIonCash:            roi,
		PerScenarioProfit:    totalScen,
	}
}
