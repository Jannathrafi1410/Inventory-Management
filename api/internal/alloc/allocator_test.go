package alloc

import (
	"encoding/binary"
	"encoding/json"
	"math"
	"os"
	"path/filepath"
	"runtime"
	"testing"
)

type cellsFile struct {
	Cells []struct {
		CellID        string  `json:"cell_id"`
		Destination   string  `json:"destination"`
		RegionGroup   string  `json:"region_group"`
		SupplierGroup string  `json:"supplier_group"`
		Segment       string  `json:"segment"`
		ContractType  string  `json:"contract_type"`
		P             float64 `json:"p"`
		CD            float64 `json:"c_d"`
		CB            float64 `json:"c_b"`
		K             float64 `json:"k"`
		R             float64 `json:"r"`
		A             float64 `json:"a"`
		G             float64 `json:"g"`
		Cap           float64 `json:"cap"`
		Group         string  `json:"group"`
	} `json:"cells"`
}

type scenarioMeta struct {
	Version string   `json:"version"`
	S       int      `json:"s"`
	N       int      `json:"n"`
	CellIDs []string `json:"cell_ids"`
}

type expectedCase struct {
	Name            string             `json:"name"`
	Budget          float64            `json:"budget"`
	LockDays        int                `json:"lock_days"`
	RAnnual         float64            `json:"r_annual"`
	ReserveFraction float64            `json:"reserve_fraction"`
	GroupCaps       map[string]float64 `json:"group_caps"`
	LambdaStar      float64            `json:"lambda_star"`
	Spend           float64            `json:"spend"`
	Quantities      []float64          `json:"quantities"`
	Fractiles       []float64          `json:"fractiles"`
}

type expectedFile struct {
	Cases []expectedCase `json:"cases"`
}

func repoRoot(t *testing.T) string {
	t.Helper()
	_, thisFile, _, ok := runtime.Caller(0)
	if !ok {
		t.Fatal("runtime.Caller failed")
	}
	// .../api/internal/alloc/allocator_test.go -> repo root
	return filepath.Clean(filepath.Join(filepath.Dir(thisFile), "..", "..", ".."))
}

func loadDemo(t *testing.T) ([]Cell, [][]float64) {
	t.Helper()
	root := repoRoot(t)
	dir := filepath.Join(root, "data", "api_demo")
	cb, err := os.ReadFile(filepath.Join(dir, "cells.json"))
	if err != nil {
		t.Skipf("demo dataset not found (%v); run python scripts/export_api_demo.py", err)
	}
	var cf cellsFile
	if err := json.Unmarshal(cb, &cf); err != nil {
		t.Fatal(err)
	}
	mb, err := os.ReadFile(filepath.Join(dir, "scenarios.meta.json"))
	if err != nil {
		t.Fatal(err)
	}
	var meta scenarioMeta
	if err := json.Unmarshal(mb, &meta); err != nil {
		t.Fatal(err)
	}
	bin, err := os.ReadFile(filepath.Join(dir, "scenarios.bin"))
	if err != nil {
		t.Fatal(err)
	}
	if len(bin) != meta.S*meta.N*8 {
		t.Fatalf("unexpected bin size %d want %d", len(bin), meta.S*meta.N*8)
	}
	sorted := make([][]float64, meta.N)
	off := 0
	for j := 0; j < meta.N; j++ {
		col := make([]float64, meta.S)
		for i := 0; i < meta.S; i++ {
			col[i] = math.Float64frombits(binary.LittleEndian.Uint64(bin[off : off+8]))
			off += 8
		}
		sorted[j] = col
	}
	byID := map[string]Cell{}
	for _, c := range cf.Cells {
		dest := c.Destination
		if dest == "" {
			dest = c.RegionGroup
		}
		sup := c.SupplierGroup
		if sup == "" {
			sup = dest
		}
		grp := c.Group
		if grp == "" {
			grp = c.RegionGroup
		}
		byID[c.CellID] = Cell{
			ID: c.CellID, P: c.P, CD: c.CD, CB: c.CB, K: c.K, R: c.R, A: c.A, G: c.G, Cap: c.Cap,
			ContractType: c.ContractType, Group: grp, Destination: dest, SupplierGroup: sup, Segment: c.Segment,
		}
	}
	cells := make([]Cell, 0, meta.N)
	if len(meta.CellIDs) == meta.N {
		for _, id := range meta.CellIDs {
			cells = append(cells, byID[id])
		}
	} else {
		for _, c := range cf.Cells {
			cells = append(cells, byID[c.CellID])
		}
	}
	return cells, sorted
}

func TestWaterfillMatchesPythonFixture(t *testing.T) {
	cells, sorted := loadDemo(t)
	root := repoRoot(t)
	eb, err := os.ReadFile(filepath.Join(root, "data", "api_demo", "expected_waterfill.json"))
	if err != nil {
		t.Fatal(err)
	}
	var ef expectedFile
	if err := json.Unmarshal(eb, &ef); err != nil {
		t.Fatal(err)
	}
	if len(ef.Cases) == 0 {
		t.Fatal("no expected cases")
	}
	for _, tc := range ef.Cases {
		t.Run(tc.Name, func(t *testing.T) {
			// Python fixture group_caps are destination/group caps in this demo
			res := Waterfill(cells, sorted, tc.Budget, tc.LockDays, tc.RAnnual, tc.ReserveFraction, tc.GroupCaps, nil)
			if math.Abs(res.LambdaStar-tc.LambdaStar) > 1e-6*math.Max(1.0, math.Abs(tc.LambdaStar)) && math.Abs(res.LambdaStar-tc.LambdaStar) > 1e-6 {
				t.Errorf("lambda: got %.12f want %.12f", res.LambdaStar, tc.LambdaStar)
			}
			if math.Abs(res.Spend-tc.Spend) > 1e-3 {
				t.Errorf("spend: got %.6f want %.6f", res.Spend, tc.Spend)
			}
			if len(res.Quantities) != len(tc.Quantities) {
				t.Fatalf("len quantities got %d want %d", len(res.Quantities), len(tc.Quantities))
			}
			for i := range res.Quantities {
				if math.Abs(res.Quantities[i]-tc.Quantities[i]) > 1.0 {
					t.Errorf("cell %d qty: got %.6f want %.6f", i, res.Quantities[i], tc.Quantities[i])
				}
			}
		})
	}
}

func BenchmarkWaterfill_50000x1000(b *testing.B) {
	// Synthetic benchmark: 50,000 cells x 1,000 scenarios (pre-sorted columns).
	// Reports real time honestly; target ~1s on demo-class hardware is environment-dependent.
	nCells := 50000
	s := 1000
	cells := make([]Cell, nCells)
	sorted := make([][]float64, nCells)
	groups := []string{"GCC", "SEA", "EU", "TUR", "IO"}
	for i := 0; i < nCells; i++ {
		g := groups[i%len(groups)]
		k := 10000.0
		if i%7 == 0 {
			k = 0
		}
		cells[i] = Cell{
			ID: "C", P: 15000, CD: 14000, CB: 10000, K: k, R: 0.7, A: 0.75, G: 100,
			Cap: 120, ContractType: "soft_release", Group: g, Destination: g, SupplierGroup: g, Segment: "leisure",
		}
		col := make([]float64, s)
		base := float64(30 + (i % 90))
		for j := 0; j < s; j++ {
			// deterministic increasing sequence (pre-sorted)
			col[j] = base + float64(j)*0.05
		}
		sorted[i] = col
	}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = Waterfill(cells, sorted, 1_000_000_000.0, 120, 0.12, 0.0, map[string]float64{"GCC": 0.4}, nil)
	}
}
