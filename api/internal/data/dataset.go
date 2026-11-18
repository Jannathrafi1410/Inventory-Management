package data

// Cell and scenario dataset loading for the Go API.
//
// Assumptions (simulator outputs under stated assumptions, not TBO real P&L):
// - Scenario binary format: column-major pre-sorted float64 little-endian, one contiguous block per cell
//   in cell_ids order, length S each; metadata in scenarios.meta.json (s,n,cell_ids,version).
// - cells.json provides cell economics/grouping derived from Python sim/optimizer cells.
// - Paths default to data/api_demo; override via env TBO_DATA_DIR or request fields where applicable.

import (
	"encoding/binary"
	"encoding/json"
	"fmt"
	"io"
	"math"
	"os"
	"path/filepath"

	"tbo-inventory-api/internal/alloc"
)

type CellsFile struct {
	Assumption string      `json:"assumption"`
	Cells      []CellInput `json:"cells"`
}

type CellInput struct {
	CellID        string  `json:"cell_id"`
	Destination   string  `json:"destination"`
	RegionGroup   string  `json:"region_group"`
	SupplierGroup string  `json:"supplier_group"`
	SourceMarket  string  `json:"source_market"`
	RatePlan      string  `json:"rate_plan"`
	ProductType   string  `json:"product_type"`
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
}

type ScenarioMeta struct {
	Version    string   `json:"version"`
	S          int      `json:"s"`
	N          int      `json:"n"`
	CellIDs    []string `json:"cell_ids"`
	DType      string   `json:"dtype"`
	Order      string   `json:"order"`
	Assumption string   `json:"assumption"`
}

type Dataset struct {
	Cells    []alloc.Cell
	Scenarios *alloc.ScenarioData
	Meta     ScenarioMeta
	CellsMeta CellsFile
	Dir      string
}

func LoadCells(path string) (CellsFile, error) {
	b, err := os.ReadFile(path)
	if err != nil {
		return CellsFile{}, err
	}
	var cf CellsFile
	if err := json.Unmarshal(b, &cf); err != nil {
		return CellsFile{}, err
	}
	if len(cf.Cells) == 0 {
		// maybe array form
		var arr []CellInput
		if err2 := json.Unmarshal(b, &arr); err2 == nil && len(arr) > 0 {
			cf.Cells = arr
		}
	}
	return cf, nil
}

func ToAllocCell(in CellInput) alloc.Cell {
	dest := in.Destination
	if dest == "" {
		dest = in.RegionGroup
	}
	if dest == "" {
		dest = in.Group
	}
	sup := in.SupplierGroup
	if sup == "" {
		sup = dest
	}
	grp := in.Group
	if grp == "" {
		grp = in.RegionGroup
	}
	return alloc.Cell{
		ID:            in.CellID,
		P:             in.P,
		CD:            in.CD,
		CB:            in.CB,
		K:             in.K,
		R:             in.R,
		A:             in.A,
		G:             in.G,
		Cap:           in.Cap,
		ContractType:  in.ContractType,
		Group:         grp,
		Destination:   dest,
		SupplierGroup: sup,
		Segment:       in.Segment,
	}
}

func LoadScenarioMeta(path string) (ScenarioMeta, error) {
	b, err := os.ReadFile(path)
	if err != nil {
		return ScenarioMeta{}, err
	}
	var m ScenarioMeta
	if err := json.Unmarshal(b, &m); err != nil {
		return ScenarioMeta{}, err
	}
	return m, nil
}

func LoadScenarioBin(path string, s, n int) ([][]float64, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()
	buf := make([]byte, 8)
	cols := make([][]float64, n)
	for j := 0; j < n; j++ {
		col := make([]float64, s)
		for i := 0; i < s; i++ {
			if _, err := io.ReadFull(f, buf); err != nil {
				return nil, fmt.Errorf("read scenario bin cell %d row %d: %w", j, i, err)
			}
			bits := binary.LittleEndian.Uint64(buf)
			col[i] = math.Float64frombits(bits)
		}
		cols[j] = col
	}
	return cols, nil
}

func LoadDataset(dir string) (*Dataset, error) {
	cellsPath := filepath.Join(dir, "cells.json")
	metaPath := filepath.Join(dir, "scenarios.meta.json")
	binPath := filepath.Join(dir, "scenarios.bin")
	cf, err := LoadCells(cellsPath)
	if err != nil {
		return nil, fmt.Errorf("load cells: %w", err)
	}
	meta, err := LoadScenarioMeta(metaPath)
	if err != nil {
		return nil, fmt.Errorf("load scenario meta: %w", err)
	}
	sorted, err := LoadScenarioBin(binPath, meta.S, meta.N)
	if err != nil {
		return nil, fmt.Errorf("load scenario bin: %w", err)
	}
	// align cells to meta.CellIDs order if provided
	byID := map[string]CellInput{}
	for _, c := range cf.Cells {
		byID[c.CellID] = c
	}
	cells := make([]alloc.Cell, 0, len(meta.CellIDs))
	if len(meta.CellIDs) > 0 {
		for _, id := range meta.CellIDs {
			if c, ok := byID[id]; ok {
				cells = append(cells, ToAllocCell(c))
			} else {
				// fallback: try match by index
			}
		}
		if len(cells) != len(meta.CellIDs) {
			cells = cells[:0]
			for _, c := range cf.Cells {
				cells = append(cells, ToAllocCell(c))
			}
		}
	} else {
		for _, c := range cf.Cells {
			cells = append(cells, ToAllocCell(c))
		}
		meta.N = len(cells)
	}
	if meta.N != len(cells) {
		// adjust to min
		if len(cells) < meta.N {
			meta.N = len(cells)
		}
		if len(sorted) > meta.N {
			sorted = sorted[:meta.N]
		}
	}
	if len(cells) > len(sorted) {
		cells = cells[:len(sorted)]
		meta.N = len(cells)
	}
	sd := &alloc.ScenarioData{Version: meta.Version, S: meta.S, N: meta.N, CellIDs: meta.CellIDs, Sorted: sorted}
	return &Dataset{Cells: cells, Scenarios: sd, Meta: meta, CellsMeta: cf, Dir: dir}, nil
}
