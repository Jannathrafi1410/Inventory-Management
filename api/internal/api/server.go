package api

// HTTP API handlers and types for /api/v1.
//
// Assumptions (simulator outputs under stated assumptions, not TBO real P&L):
// - Recommendations include Phase-6-equivalent fields from optimizer/planner.py plus destination/segment/group.
// - "expected_uplift"/"cvar95_uplift" are incremental vs no-pre-buy baseline on provided scenarios.
// - If risk.beta>0, plan uses mean-CVaR LP when a Go implementation is available? Not ported; allocator uses
//   water-filling (risk-neutral/budgeted fractile) and metrics report E[dProfit]/CVaR95 of that plan. Assumption
//   documented in response metadata; beta/alpha accepted for API contract.
// - POST /api/v1/scenarios/run (re)exports demo dataset from Python sources and reloads it by default.

import (
	"crypto/rand"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"math"
	"net/http"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strings"
	"sync"
	"time"

	"tbo-inventory-api/internal/alloc"
	"tbo-inventory-api/internal/data"
	"tbo-inventory-api/internal/store"
)

type Horizon struct {
	From string `json:"from"`
	To   string `json:"to"`
}

type Risk struct {
	Beta  float64 `json:"beta"`
	Alpha float64 `json:"alpha"`
}

type Constraints struct {
	MaxSharePerDestination float64  `json:"max_share_per_destination"`
	MaxSharePerSupplier    float64  `json:"max_share_per_supplier"`
	MinFillRate            float64  `json:"min_fill_rate"`
	AllowedStructures      []string `json:"allowed_structures"`
	ReserveFrac            float64  `json:"reserve_frac"`
	ExplorationFrac        float64  `json:"exploration_frac"`
}

type PlanRequest struct {
	Budget      float64     `json:"budget"`
	Currency    string      `json:"currency"`
	Horizon     Horizon     `json:"horizon"`
	Risk        Risk        `json:"risk"`
	Constraints Constraints `json:"constraints"`
	Segments    []string    `json:"segments"`
	// optional overrides
	DatasetDir string `json:"dataset_dir,omitempty"`
}

type Recommendation struct {
	CellID       string   `json:"cell_id"`
	Action       string   `json:"action"`
	Structure    string   `json:"structure"`
	Units        float64  `json:"units"`
	Cash         float64  `json:"cash"`
	UnitNet      float64  `json:"unit_net"`
	SellPrice    float64  `json:"sell_price"`
	Fractile     float64  `json:"fractile"`
	DeltaM       float64  `json:"delta_m"`
	CO           float64  `json:"c_o"`
	DemandP10    float64  `json:"demand_p10"`
	DemandP50    float64  `json:"demand_p50"`
	DemandP90    float64  `json:"demand_p90"`
	SelloutProb  float64  `json:"sellout_prob"`
	Why          []string `json:"why"`
	Destination  string   `json:"destination,omitempty"`
	Segment      string   `json:"segment,omitempty"`
	Group        string   `json:"group,omitempty"`
}

type PlanResponse struct {
	PlanID               string           `json:"plan_id"`
	Lambda               float64          `json:"lambda"`
	CashDeployed         float64          `json:"cash_deployed"`
	Reserve              float64          `json:"reserve"`
	ExpectedUplift       float64          `json:"expected_uplift"`
	StructuralUplift     float64          `json:"structural_uplift"`
	CVaR95Uplift         float64          `json:"cvar95_uplift"`
	ROIonCash            float64          `json:"roi_on_cash"`
	ExpectedCashSpoilage float64          `json:"expected_cash_spoilage"`
	FillRateUpliftPP     float64          `json:"fill_rate_uplift_pp"`
	Recommendations      []Recommendation `json:"recommendations"`
	// metadata
	SolveTimeMs   float64 `json:"solve_time_ms,omitempty"`
	DatasetVersion string `json:"dataset_version,omitempty"`
	Assumption    string  `json:"assumption,omitempty"`
}

type ScenarioRunRequest struct {
	DatasetDir string `json:"dataset_dir,omitempty"`
	Script     string `json:"script,omitempty"`
}

type Server struct {
	mu      sync.RWMutex
	ds      *data.Dataset
	store   *store.Store
	dataDir string
	root    string
}

func NewServer(root, dataDir string, st *store.Store) (*Server, error) {
	s := &Server{store: st, dataDir: dataDir, root: root}
	if _, err := os.Stat(filepath.Join(dataDir, "cells.json")); err == nil {
		_ = s.reload(dataDir)
	}
	return s, nil
}

func (s *Server) reload(dir string) error {
	ds, err := data.LoadDataset(dir)
	if err != nil {
		return err
	}
	s.mu.Lock()
	s.ds = ds
	s.dataDir = dir
	s.mu.Unlock()
	return nil
}

func (s *Server) dataset() *data.Dataset {
	s.mu.RLock()
	defer s.mu.RUnlock()
	return s.ds
}

func newID() string {
	b := make([]byte, 8)
	_, _ = rand.Read(b)
	return "plan_" + hex.EncodeToString(b)
}

func containsStr(list []string, v string) bool {
	if len(list) == 0 {
		return true
	}
	for _, x := range list {
		if strings.EqualFold(x, v) {
			return true
		}
	}
	return false
}

func filterCells(ds *data.Dataset, req PlanRequest) ([]alloc.Cell, *alloc.ScenarioData, []int) {
	idxs := []int{}
	cells := []alloc.Cell{}
	for i, c := range ds.Cells {
		if len(req.Segments) > 0 && !containsStr(req.Segments, c.Segment) {
			continue
		}
		if len(req.Constraints.AllowedStructures) > 0 && !containsStr(req.Constraints.AllowedStructures, c.ContractType) {
			continue
		}
		idxs = append(idxs, i)
		cells = append(cells, c)
	}
	// subset scenarios
	sd := &alloc.ScenarioData{Version: ds.Scenarios.Version, S: ds.Scenarios.S, N: len(idxs), Sorted: make([][]float64, len(idxs))}
	ids := make([]string, len(idxs))
	for j, i := range idxs {
		ids[j] = ds.Cells[i].ID
		sd.Sorted[j] = ds.Scenarios.Sorted[i]
	}
	sd.CellIDs = ids
	return cells, sd, idxs
}

func whyBullets(c alloc.Cell, q float64, dcol []float64, action string, fractile, lambda float64, lockDays int) []string {
	p10, p50, p90 := alloc.QuantileFromSorted(dcol, 0.10), alloc.QuantileFromSorted(dcol, 0.50), alloc.QuantileFromSorted(dcol, 0.90)
	carry := c.Carry(lockDays, 0.12)
	dm := c.DeltaM()
	co := c.CO()
	whys := []string{}
	switch action {
	case "DONT_BUY":
		whys = append(whys, fmt.Sprintf("DONT_BUY: delta_m=%.0f is not attractive vs dynamic fallback/carry.", dm))
		return whys
	case "WAIT":
		whys = append(whys, fmt.Sprintf("WAIT: defer commitment (demand p10/p50/p90=%.0f/%.0f/%.0f) to preserve cash flexibility.", p10, p50, p90))
		if dm > 0 {
			whys = append(whys, fmt.Sprintf("Economics positive (delta_m=%.0f) but not committing now under current constraints.", dm))
		}
		return whys
	case "NEGOTIATE_ALLOTMENT":
		whys = append(whys, fmt.Sprintf("NEGOTIATE_ALLOTMENT: positive edge (delta_m=%.0f) but cash-funded buy not taken now.", dm))
		whys = append(whys, fmt.Sprintf("Seek softer terms (lower k/higher r) given c_o=%.0f and demand p50=%.0f.", co, p50))
		return whys
	case "RELEASE":
		whys = append(whys, "RELEASE: do not hold/commit these units now given current economics and demand risk.")
		return whys
	}
	if c.K == 0 {
		whys = append(whys, "COMMIT: zero cash at risk (k=0, free option) so taking available cap is optimal.")
	} else {
		whys = append(whys, fmt.Sprintf("Sold at the %.0fth demand percentile, not the mean, because cash is at risk (c_o=%.0f, r=%.2f).", fractile*100, co, c.R))
		whys = append(whys, fmt.Sprintf("Per-unit edge delta_m=%.0f net of fallback, with carry=%.0f over %dd lock.", dm, carry, lockDays))
	}
	if lambda > 0 && c.K > 0 {
		ratio := dm / math.Max(c.K, 1e-9)
		whys = append(whys, fmt.Sprintf("Budget-binding allocation (lambda=%.4f); selected on delta_m/k=%.4f.", lambda, ratio))
	}
	sp := 0.0
	n := len(dcol)
	if n > 0 {
		cnt := 0
		for _, d := range dcol {
			if d > q {
				cnt++
			}
		}
		sp = float64(cnt) / float64(n)
	}
	whys = append(whys, fmt.Sprintf("Demand p10/p50/p90=%.0f/%.0f/%.0f; sellout probability at q=%.1f is %.1f%%.", p10, p50, p90, q, sp*100))
	if len(whys) > 5 {
		whys = whys[:5]
	}
	return whys
}

func computePlan(ds *data.Dataset, req PlanRequest) (PlanResponse, []alloc.Cell, *alloc.ScenarioData, alloc.Result, error) {
	cells, sd, _ := filterCells(ds, req)
	if len(cells) == 0 {
		return PlanResponse{}, nil, nil, alloc.Result{}, fmt.Errorf("no cells after filtering")
	}
	lockDays := 120
	rAnnual := 0.12
	reserveFrac := req.Constraints.ReserveFrac
	destCaps := map[string]float64{}
	supCaps := map[string]float64{}
	if req.Constraints.MaxSharePerDestination > 0 {
		// apply as global cap fraction per destination group key
		// handled by applying same fraction to every destination seen
		// easier: pass special? build per-group map
		destGroups := map[string]bool{}
		for _, c := range cells {
			destGroups[c.Destination] = true
		}
		for g := range destGroups {
			destCaps[g] = req.Constraints.MaxSharePerDestination
		}
	}
	if req.Constraints.MaxSharePerSupplier > 0 {
		supGroups := map[string]bool{}
		for _, c := range cells {
			supGroups[c.SupplierGroup] = true
		}
		for g := range supGroups {
			supCaps[g] = req.Constraints.MaxSharePerSupplier
		}
	}
	start := time.Now()
	res := alloc.Waterfill(cells, sd.Sorted, req.Budget, lockDays, rAnnual, reserveFrac, destCaps, supCaps)
	elapsed := time.Since(start).Seconds() * 1000.0

	alpha := req.Risk.Alpha
	if alpha <= 0 || alpha >= 1 {
		alpha = 0.95
	}
	metrics := alloc.EvaluatePortfolio(cells, res.Quantities, sd.Sorted, lockDays, rAnnual, alpha)
	reserve := req.Budget * reserveFrac
	if reserveFrac < 0 {
		reserve = 0
	}
	if reserveFrac > 0.95 {
		reserve = req.Budget * 0.95
	}

	recs := make([]Recommendation, 0, len(cells))
	for i, c := range cells {
		q := res.Quantities[i]
		fr := 0.0
		if i < len(res.Diagnostics) {
			fr = res.Diagnostics[i].FractileUsed
		}
		p10, p50, p90 := sd.Quantiles(i)
		sp := sd.SelloutProb(i, q)
		dm := c.DeltaM()
		co := c.CO()
		dcol := sd.Sorted[i]
		action := "WAIT"
		if q > 1e-9 {
			action = "COMMIT"
		} else {
			if dm <= 0.0 {
				action = "DONT_BUY"
			} else if c.K <= 0.0 {
				action = "WAIT"
			} else {
				action = "NEGOTIATE_ALLOTMENT"
			}
		}
		// respect min_fill_rate if requested (soft): if committing but sellout prob very low vs target, suggest WAIT
		if req.Constraints.MinFillRate > 0 && action == "COMMIT" {
			expFill := 1.0 - sp
			if expFill < req.Constraints.MinFillRate && c.K > 0 {
				// keep commit but note in why handled below; action remains commit per allocation
			}
		}
		whys := whyBullets(c, q, dcol, action, fr, res.LambdaStar, lockDays)
		if req.Constraints.MinFillRate > 0 && action == "COMMIT" && c.K > 0 {
			expFill := 1.0 - sp
			if expFill < req.Constraints.MinFillRate {
				whys = append([]string{fmt.Sprintf("Note: expected fill %.1f%% below requested min_fill_rate %.1f%% (simulator assumption).", expFill*100, req.Constraints.MinFillRate*100)}, whys...)
				if len(whys) > 5 {
					whys = whys[:5]
				}
			}
		}
		recs = append(recs, Recommendation{
			CellID: c.ID, Action: action, Structure: c.ContractType, Units: q, Cash: c.K * q,
			UnitNet: c.CB, SellPrice: c.P, Fractile: fr, DeltaM: dm, CO: co,
			DemandP10: p10, DemandP50: p50, DemandP90: p90, SelloutProb: sp, Why: whys,
			Destination: c.Destination, Segment: c.Segment, Group: c.Group,
		})
	}
	resp := PlanResponse{
		PlanID:               newID(),
		Lambda:               res.LambdaStar,
		CashDeployed:         res.Spend,
		Reserve:              reserve,
		ExpectedUplift:       metrics.ExpectedUplift,
		StructuralUplift:     metrics.StructuralUplift,
		CVaR95Uplift:         metrics.CVaR95Uplift,
		ROIonCash:            metrics.ROIonCash,
		ExpectedCashSpoilage: metrics.ExpectedCashSpoilage,
		FillRateUpliftPP:     metrics.FillRateUpliftPP,
		Recommendations:      recs,
		SolveTimeMs:          elapsed,
		DatasetVersion:       sd.Version,
		Assumption:           "Simulator outputs under stated assumptions, not TBO real P&L. Allocator: water-filling port (60 bisection iterations). risk.beta accepted in request; metrics report plan CVaR95 uplift.",
	}
	return resp, cells, sd, res, nil
}

func writeJSON(w http.ResponseWriter, status int, v any) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	_ = json.NewEncoder(w).Encode(v)
}

func (s *Server) HandlePlan(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		writeJSON(w, http.StatusMethodNotAllowed, map[string]string{"error": "method not allowed"})
		return
	}
	var req PlanRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		writeJSON(w, http.StatusBadRequest, map[string]string{"error": "invalid json: " + err.Error()})
		return
	}
	if req.Budget <= 0 {
		writeJSON(w, http.StatusBadRequest, map[string]string{"error": "budget must be > 0"})
		return
	}
	if req.Currency == "" {
		req.Currency = "INR"
	}
	if req.Constraints.ReserveFrac < 0 {
		req.Constraints.ReserveFrac = 0
	}
	ds := s.dataset()
	if req.DatasetDir != "" {
		if err := s.reload(req.DatasetDir); err != nil {
			writeJSON(w, http.StatusBadRequest, map[string]string{"error": "failed to load dataset: " + err.Error()})
			return
		}
		ds = s.dataset()
	}
	if ds == nil {
		writeJSON(w, http.StatusServiceUnavailable, map[string]string{"error": "no dataset loaded; run POST /api/v1/scenarios/run first"})
		return
	}
	resp, _, _, _, err := computePlan(ds, req)
	if err != nil {
		writeJSON(w, http.StatusBadRequest, map[string]string{"error": err.Error()})
		return
	}
	if s.store != nil {
		_, _ = s.store.Save(resp.PlanID, req, resp)
	}
	writeJSON(w, http.StatusOK, resp)
}

func (s *Server) HandleGetPlan(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodGet {
		writeJSON(w, http.StatusMethodNotAllowed, map[string]string{"error": "method not allowed"})
		return
	}
	id := strings.TrimPrefix(r.URL.Path, "/api/v1/plans/")
	if id == "" || strings.Contains(id, "/") {
		writeJSON(w, http.StatusBadRequest, map[string]string{"error": "missing plan id"})
		return
	}
	if s.store == nil {
		writeJSON(w, http.StatusServiceUnavailable, map[string]string{"error": "plan store not configured"})
		return
	}
	sp, err := s.store.Get(id)
	if err != nil {
		writeJSON(w, http.StatusNotFound, map[string]string{"error": "plan not found"})
		return
	}
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(http.StatusOK)
	_ = json.NewEncoder(w).Encode(sp)
}

func (s *Server) HandleScenariosRun(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		writeJSON(w, http.StatusMethodNotAllowed, map[string]string{"error": "method not allowed"})
		return
	}
	var req ScenarioRunRequest
	_ = json.NewDecoder(r.Body).Decode(&req)
	dir := req.DatasetDir
	if dir == "" {
		dir = filepath.Join(s.root, "data", "api_demo")
	}
	script := req.Script
	if script == "" {
		script = filepath.Join(s.root, "scripts", "export_api_demo.py")
	}
	// Run python export if script exists
	start := time.Now()
	msg := ""
	if _, err := os.Stat(script); err == nil {
		py := "python"
		if runtime.GOOS == "windows" {
			// try python
		}
		cmd := exec.Command(py, script)
		cmd.Dir = s.root
		out, err := cmd.CombinedOutput()
		msg = string(out)
		if err != nil {
			writeJSON(w, http.StatusInternalServerError, map[string]any{"error": "scenario export failed", "detail": err.Error(), "output": msg})
			return
		}
	}
	if err := s.reload(dir); err != nil {
		writeJSON(w, http.StatusBadRequest, map[string]any{"error": "failed to load dataset", "detail": err.Error(), "output": msg})
		return
	}
	elapsed := time.Since(start).Seconds() * 1000.0
	ds := s.dataset()
	writeJSON(w, http.StatusOK, map[string]any{
		"status": "ok", "dataset_dir": s.dataDir, "version": ds.Scenarios.Version,
		"s": ds.Scenarios.S, "n": ds.Scenarios.N, "elapsed_ms": elapsed, "output": strings.TrimSpace(msg),
		"assumption": "Demo scenario export uses seeded LogNormal draws from sim candidates; simulator outputs under stated assumptions.",
	})
}

func (s *Server) HandleHealthz(w http.ResponseWriter, r *http.Request) {
	ds := s.dataset()
	loaded := ds != nil
	ver := ""
	if loaded {
		ver = ds.Scenarios.Version
	}
	writeJSON(w, http.StatusOK, map[string]any{"status": "ok", "service": "tbo-inventory-api", "dataset_loaded": loaded, "dataset_version": ver})
}

func (s *Server) Routes() http.Handler {
	mux := http.NewServeMux()
	mux.HandleFunc("/healthz", s.HandleHealthz)
	mux.HandleFunc("/health", s.HandleHealthz)
	mux.HandleFunc("/api/v1/plan", s.HandlePlan)
	mux.HandleFunc("/api/v1/plans/", s.HandleGetPlan)
	mux.HandleFunc("/api/v1/scenarios/run", s.HandleScenariosRun)
	return mux
}
