package alloc

// ScenarioData holds pre-sorted empirical demand scenarios per cell [nCells][S].
// Scenarios are stored sorted ascending per cell for binary-search quantile lookup.

type ScenarioData struct {
	Version string      `json:"version"`
	S       int         `json:"s"`
	N       int         `json:"n"`
	CellIDs []string    `json:"cell_ids"`
	Sorted  [][]float64 `json:"-"` // [n][s]
}

// Quantiles returns p10, p50, p90 from pre-sorted column.
func (sd *ScenarioData) Quantiles(cellIndex int) (p10, p50, p90 float64) {
	if cellIndex < 0 || cellIndex >= len(sd.Sorted) || len(sd.Sorted[cellIndex]) == 0 {
		return 0, 0, 0
	}
	col := sd.Sorted[cellIndex]
	return QuantileFromSorted(col, 0.10), QuantileFromSorted(col, 0.50), QuantileFromSorted(col, 0.90)
}

// SelloutProb returns P(D > q) from pre-sorted column.
func (sd *ScenarioData) SelloutProb(cellIndex int, q float64) float64 {
	if cellIndex < 0 || cellIndex >= len(sd.Sorted) {
		return 0
	}
	col := sd.Sorted[cellIndex]
	n := len(col)
	if n == 0 {
		return 0
	}
	// count elements > q
	// binary search for first index with value > q
	lo, hi := 0, n
	for lo < hi {
		mid := (lo + hi) / 2
		if col[mid] <= q {
			lo = mid + 1
		} else {
			hi = mid
		}
	}
	return float64(n-lo) / float64(n)
}
