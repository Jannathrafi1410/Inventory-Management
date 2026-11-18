package alloc

// Water-filling allocator ported from Python optimizer/waterfill.py.
//
// Assumptions (simulator outputs under stated assumptions, not TBO real P&L):
// - Demand columns are pre-sorted ascending per cell (empirical CDF).
// - Quantile lookup uses ceil(f*S)-1 index on pre-sorted samples (clipped), equivalent to Python.
// - Bisection on lambda runs for 60 iterations after finding an upper bound by doubling.
// - carry = k * rAnnual * lockDays / 365. delta_m, c_o, fractiles follow core math exactly.
// - Group caps: values in (0,1] treated as fraction of effective budget; values >1 as absolute cash.
// - Effective budget = budget * (1 - reserveFraction). k==0 (free option) taken to cap if delta_m-carry>0.

import (
	"math"
	"sort"
)

// Cell is a pre-buy decision cell with economics and grouping.
type Cell struct {
	ID            string  `json:"cell_id"`
	P             float64 `json:"p"`
	CD            float64 `json:"c_d"`
	CB            float64 `json:"c_b"`
	K             float64 `json:"k"`
	R             float64 `json:"r"`
	A             float64 `json:"a"`
	G             float64 `json:"g"`
	Cap           float64 `json:"cap"`
	ContractType  string  `json:"contract_type"`
	Group         string  `json:"group"`          // destination/concentration group for legacy caps
	Destination   string  `json:"destination"`   // per-destination exposure group
	SupplierGroup string  `json:"supplier_group"` // per-supplier exposure group
	Segment       string  `json:"segment"`
}

func (c Cell) DeltaM() float64 {
	return (c.P - c.CB) - c.A*(c.P-c.CD) + (1.0-c.A)*c.G
}

func (c Cell) CO() float64 { return c.K * (1.0 - c.R) }

func (c Cell) Carry(lockDays int, rAnnual float64) float64 {
	return c.K * rAnnual * float64(lockDays) / 365.0
}

// CellDiagnostic holds per-cell allocation diagnostics.
type CellDiagnostic struct {
	CellID       string  `json:"cell_id"`
	FractileUsed float64 `json:"fractile_used"`
	BindingCap   bool    `json:"binding_cap"`
	DeltaMOverK  float64 `json:"delta_m_over_k"`
	Quantity     float64 `json:"quantity"`
	Spend        float64 `json:"spend"`
	Group        string  `json:"group"`
	DeltaM       float64 `json:"delta_m"`
	Carry        float64 `json:"carry"`
	CO           float64 `json:"c_o"`
	K            float64 `json:"k"`
}

// Result is the output of budgeted water-filling.
type Result struct {
	Quantities  []float64        `json:"quantities"`
	LambdaStar  float64          `json:"lambda_star"`
	Spend       float64          `json:"spend"`
	Diagnostics []CellDiagnostic `json:"diagnostics"`
}

func clip01(x float64) float64 {
	if x < 0 {
		return 0
	}
	if x > 1 {
		return 1
	}
	return x
}

// QuantileFromSorted returns demand quantile for fractile using pre-sorted scenarios.
// Implements ceil(f*S)-1 with clipping; if f<=0 returns first element.
func QuantileFromSorted(sortedCol []float64, fractile float64) float64 {
	s := len(sortedCol)
	if s == 0 {
		return 0
	}
	f := clip01(fractile)
	if f <= 0.0 {
		return sortedCol[0]
	}
	idx := int(math.Ceil(f*float64(s))) - 1
	if idx < 0 {
		idx = 0
	}
	if idx >= s {
		idx = s - 1
	}
	return sortedCol[idx]
}

func normalizeGroupCaps(groupCaps map[string]float64, effectiveBudget float64) map[string]float64 {
	out := map[string]float64{}
	if len(groupCaps) == 0 {
		return out
	}
	for g, v := range groupCaps {
		if v <= 1.0 && v > 0 {
			out[g] = v * effectiveBudget
		} else if v <= 0 {
			out[g] = 0
		} else {
			out[g] = v
		}
	}
	return out
}

// groupKey returns the cap group key for a cell depending on cap mode.
func groupKey(c Cell, mode string) string {
	switch mode {
	case "destination":
		if c.Destination != "" {
			return c.Destination
		}
		return c.Group
	case "supplier":
		if c.SupplierGroup != "" {
			return c.SupplierGroup
		}
		return c.Group
	default:
		return c.Group
	}
}

func applyGroupCaps(cells []Cell, q []float64, effectiveBudget float64, groupCaps map[string]float64, mode string) []float64 {
	capsAbs := normalizeGroupCaps(groupCaps, effectiveBudget)
	if len(capsAbs) == 0 {
		out := make([]float64, len(q))
		copy(out, q)
		return out
	}
	qAdj := make([]float64, len(q))
	copy(qAdj, q)
	groups := map[string][]int{}
	for i, c := range cells {
		keys := []string{groupKey(c, mode), c.Group, c.Destination, c.SupplierGroup}
		seenKey := map[string]bool{}
		for _, k := range keys {
			if k == "" || seenKey[k] {
				continue
			}
			seenKey[k] = true
			groups[k] = append(groups[k], i)
		}
	}
	for g, limit := range capsAbs {
		idxsAll := groups[g]
		seenIdx := map[int]bool{}
		idxs := make([]int, 0, len(idxsAll))
		for _, i := range idxsAll {
			if seenIdx[i] || cells[i].K <= 0 {
				continue
			}
			seenIdx[i] = true
			idxs = append(idxs, i)
		}
		if len(idxs) == 0 {
			continue
		}
		spendG := 0.0
		for _, i := range idxs {
			spendG += cells[i].K * qAdj[i]
		}
		if spendG <= limit+1e-9 {
			continue
		}
		sort.Slice(idxs, func(a, b int) bool {
			ia, ib := idxs[a], idxs[b]
			ra := math.Inf(1)
			rb := math.Inf(1)
			if cells[ia].K > 0 {
				ra = cells[ia].DeltaM() / cells[ia].K
			}
			if cells[ib].K > 0 {
				rb = cells[ib].DeltaM() / cells[ib].K
			}
			return ra < rb
		})
		for _, i := range idxs {
			if spendG <= limit+1e-9 {
				break
			}
			k := cells[i].K
			maxReducible := k * qAdj[i]
			need := spendG - limit
			reduceCash := math.Min(maxReducible, need)
			if k > 0 {
				qAdj[i] = math.Max(0.0, qAdj[i]-reduceCash/k)
				spendG -= reduceCash
			}
		}
	}
	return qAdj
}

func applyAllGroupCaps(cells []Cell, q []float64, effectiveBudget float64, destinationCaps, supplierCaps map[string]float64) []float64 {
	out := applyGroupCaps(cells, q, effectiveBudget, destinationCaps, "destination")
	out = applyGroupCaps(cells, out, effectiveBudget, supplierCaps, "supplier")
	return out
}

func allocateForLambda(cells []Cell, sortedDemand [][]float64, lam float64, effectiveBudget float64, lockDays int, rAnnual float64, destinationCaps, supplierCaps map[string]float64) ([]float64, float64, []float64) {
	n := len(cells)
	q := make([]float64, n)
	fractiles := make([]float64, n)
	for i, c := range cells {
		dm := c.DeltaM()
		co := c.CO()
		car := c.Carry(lockDays, rAnnual)
		var f float64
		if c.K <= 0.0 {
			if dm-car > 0.0 {
				f = 1.0
			} else {
				f = 0.0
			}
		} else {
			num := dm - car - lam*c.K
			if co == 0.0 {
				if num > 0.0 {
					f = 1.0
				} else {
					f = 0.0
				}
			} else if dm <= 0.0 || num <= 0.0 {
				f = 0.0
			} else {
				f = clip01(num / (dm + co))
			}
		}
		fractiles[i] = f
		qStar := 0.0
		if i < len(sortedDemand) {
			qStar = QuantileFromSorted(sortedDemand[i], f)
		}
		q[i] = math.Min(c.Cap, math.Max(qStar, 0.0))
	}
	q = applyAllGroupCaps(cells, q, effectiveBudget, destinationCaps, supplierCaps)
	spend := 0.0
	for i := range cells {
		spend += cells[i].K * q[i]
	}
	return q, spend, fractiles
}

func totalSpend(cells []Cell, q []float64) float64 {
	s := 0.0
	for i := range cells {
		s += cells[i].K * q[i]
	}
	return s
}

// Waterfill allocates quantities under a cash budget via bisection on lambda (60 iterations).
// sortedDemand is [nCells][S] with each column pre-sorted ascending.
func Waterfill(cells []Cell, sortedDemand [][]float64, budget float64, lockDays int, rAnnual float64, reserveFraction float64, destinationCaps, supplierCaps map[string]float64) Result {
	n := len(cells)
	if lockDays <= 0 {
		lockDays = 120
	}
	if rAnnual <= 0 {
		rAnnual = 0.12
	}
	if reserveFraction < 0 {
		reserveFraction = 0
	}
	if reserveFraction > 0.95 {
		reserveFraction = 0.95
	}
	effectiveBudget := budget * (1.0 - reserveFraction)
	if effectiveBudget < 0 {
		effectiveBudget = 0
	}

	q0, spend0, f0 := allocateForLambda(cells, sortedDemand, 0.0, effectiveBudget, lockDays, rAnnual, destinationCaps, supplierCaps)
	var lam float64
	q := q0
	spend := spend0
	_ = f0

	if spend0 <= effectiveBudget+1e-9 {
		lam = 0.0
	} else {
		lamLo := 0.0
		lamHi := 1e-8
		qHi, spendHi, _ := allocateForLambda(cells, sortedDemand, lamHi, effectiveBudget, lockDays, rAnnual, destinationCaps, supplierCaps)
		for i := 0; i < 200; i++ {
			qHi, spendHi, _ = allocateForLambda(cells, sortedDemand, lamHi, effectiveBudget, lockDays, rAnnual, destinationCaps, supplierCaps)
			if spendHi <= effectiveBudget+1e-9 {
				break
			}
			lamLo = lamHi
			lamHi *= 2.0
		}
		q = qHi
		spend = spendHi
		lam = lamHi
		for i := 0; i < 60; i++ {
			lamMid := 0.5 * (lamLo + lamHi)
			qMid, spendMid, _ := allocateForLambda(cells, sortedDemand, lamMid, effectiveBudget, lockDays, rAnnual, destinationCaps, supplierCaps)
			if spendMid > effectiveBudget+1e-9 {
				lamLo = lamMid
			} else {
				lamHi = lamMid
				q = qMid
				spend = spendMid
			}
			lam = lamHi
			tol := 1e-6 * math.Max(1.0, lamHi)
			if math.Abs(lamHi-lamLo) <= tol {
				break
			}
		}
		if spend > effectiveBudget+1e-6 {
			for i := 0; i < 200; i++ {
				lamHi = math.Max(lamHi*2.0, lamHi+1e-12)
				q, spend, _ = allocateForLambda(cells, sortedDemand, lamHi, effectiveBudget, lockDays, rAnnual, destinationCaps, supplierCaps)
				lam = lamHi
				if spend <= effectiveBudget+1e-6 {
					break
				}
			}
		}
		if spend > effectiveBudget+1e-6 && spend > 0 {
			scale := effectiveBudget / spend
			q = append([]float64(nil), q...)
			for i, c := range cells {
				if c.K > 0 {
					q[i] *= scale
				}
			}
			spend = totalSpend(cells, q)
		}
	}

	diags := make([]CellDiagnostic, n)
	for i, c := range cells {
		dm := c.DeltaM()
		car := c.Carry(lockDays, rAnnual)
		co := c.CO()
		dmOverK := math.Inf(1)
		if c.K != 0 {
			dmOverK = dm / c.K
		}
		var fUsed float64
		if c.K <= 0.0 {
			if dm-car > 0.0 {
				fUsed = 1.0
			} else {
				fUsed = 0.0
			}
		} else {
			num := dm - car - lam*c.K
			if co == 0.0 {
				if num > 0.0 {
					fUsed = 1.0
				} else {
					fUsed = 0.0
				}
			} else if dm <= 0.0 || num <= 0.0 {
				fUsed = 0.0
			} else {
				fUsed = clip01(num / (dm + co))
			}
		}
		binding := q[i] >= c.Cap-1e-9 && fUsed > 0.0
		diags[i] = CellDiagnostic{
			CellID:       c.ID,
			FractileUsed: fUsed,
			BindingCap:   binding,
			DeltaMOverK:  dmOverK,
			Quantity:     q[i],
			Spend:        c.K * q[i],
			Group:        c.Group,
			DeltaM:       dm,
			Carry:        car,
			CO:           co,
			K:            c.K,
		}
	}
	return Result{Quantities: q, LambdaStar: lam, Spend: spend, Diagnostics: diags}
}
