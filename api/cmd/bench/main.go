package main

// Synthetic benchmark/debug runner for large water-filling instances.
// Usage: go run ./cmd/bench -cells 50000 -scenarios 1000

import (
	"flag"
	"fmt"
	"time"

	"tbo-inventory-api/internal/alloc"
)

func main() {
	nCells := flag.Int("cells", 50000, "number of cells")
	s := flag.Int("scenarios", 1000, "scenarios per cell")
	budget := flag.Float64("budget", 1_000_000_000, "budget")
	flag.Parse()

	cells := make([]alloc.Cell, *nCells)
	sorted := make([][]float64, *nCells)
	groups := []string{"GCC", "SEA", "EU", "TUR", "IO"}
	for i := 0; i < *nCells; i++ {
		g := groups[i%len(groups)]
		k := 10000.0
		if i%7 == 0 {
			k = 0
		}
		cells[i] = alloc.Cell{ID: fmt.Sprintf("C%d", i), P: 15000, CD: 14000, CB: 10000, K: k, R: 0.7, A: 0.75, G: 100, Cap: 120, ContractType: "soft_release", Group: g, Destination: g, SupplierGroup: g, Segment: "leisure"}
		col := make([]float64, *s)
		base := float64(30 + (i % 90))
		for j := 0; j < *s; j++ {
			col[j] = base + float64(j)*0.05
		}
		sorted[i] = col
	}
	start := time.Now()
	res := alloc.Waterfill(cells, sorted, *budget, 120, 0.12, 0.0, map[string]float64{"GCC": 0.4}, nil)
	elapsed := time.Since(start)
	fmt.Printf("cells=%d scenarios=%d elapsed_ms=%.3f lambda=%.8f spend=%.2f\n", *nCells, *s, float64(elapsed.Microseconds())/1000.0, res.LambdaStar, res.Spend)
}
