package api

import (
	"bytes"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"path/filepath"
	"runtime"
	"testing"

	"tbo-inventory-api/internal/store"
)

func repoRoot(t *testing.T) string {
	t.Helper()
	_, thisFile, _, ok := runtime.Caller(0)
	if !ok {
		t.Fatal("runtime.Caller failed")
	}
	return filepath.Clean(filepath.Join(filepath.Dir(thisFile), "..", "..", ".."))
}

func newTestServer(t *testing.T) *Server {
	t.Helper()
	root := repoRoot(t)
	dataDir := filepath.Join(root, "data", "api_demo")
	storeDir := t.TempDir()
	st, err := store.New(storeDir)
	if err != nil {
		t.Fatal(err)
	}
	srv, err := NewServer(root, dataDir, st)
	if err != nil {
		t.Fatal(err)
	}
	if srv.dataset() == nil {
		t.Skip("demo dataset not loaded; run python scripts/export_api_demo.py")
	}
	return srv
}

func TestHealthz(t *testing.T) {
	srv := newTestServer(t)
	req := httptest.NewRequest(http.MethodGet, "/healthz", nil)
	w := httptest.NewRecorder()
	srv.Routes().ServeHTTP(w, req)
	if w.Code != http.StatusOK {
		t.Fatalf("status %d body %s", w.Code, w.Body.String())
	}
	var resp map[string]any
	if err := json.Unmarshal(w.Body.Bytes(), &resp); err != nil {
		t.Fatal(err)
	}
	if resp["status"] != "ok" {
		t.Fatalf("unexpected resp: %#v", resp)
	}
}

func TestPlanAndGetPlan(t *testing.T) {
	srv := newTestServer(t)
	body := map[string]any{
		"budget": 10000000.0, "currency": "INR",
		"horizon": map[string]string{"from": "2026-12-01", "to": "2026-12-31"},
		"risk":    map[string]any{"beta": 0.5, "alpha": 0.95},
		"constraints": map[string]any{
			"max_share_per_destination": 0.5,
			"max_share_per_supplier":    0.5,
			"min_fill_rate":             0.0,
			"allowed_structures":        []string{},
			"reserve_frac":              0.0,
			"exploration_frac":          0.0,
		},
		"segments": []string{},
	}
	b, _ := json.Marshal(body)
	req := httptest.NewRequest(http.MethodPost, "/api/v1/plan", bytes.NewReader(b))
	w := httptest.NewRecorder()
	srv.Routes().ServeHTTP(w, req)
	if w.Code != http.StatusOK {
		t.Fatalf("plan status %d body %s", w.Code, w.Body.String())
	}
	var planResp map[string]any
	if err := json.Unmarshal(w.Body.Bytes(), &planResp); err != nil {
		t.Fatal(err)
	}
	id, _ := planResp["plan_id"].(string)
	if id == "" {
		t.Fatal("missing plan_id")
	}
	if _, ok := planResp["lambda"]; !ok {
		t.Fatal("missing lambda")
	}
	if _, ok := planResp["recommendations"]; !ok {
		t.Fatal("missing recommendations")
	}
	// GET plan
	req2 := httptest.NewRequest(http.MethodGet, "/api/v1/plans/"+id, nil)
	w2 := httptest.NewRecorder()
	srv.Routes().ServeHTTP(w2, req2)
	if w2.Code != http.StatusOK {
		t.Fatalf("get plan status %d body %s", w2.Code, w2.Body.String())
	}
	var stored map[string]any
	if err := json.Unmarshal(w2.Body.Bytes(), &stored); err != nil {
		t.Fatal(err)
	}
	if stored["plan_id"] != id {
		t.Fatalf("stored id mismatch: %#v", stored["plan_id"])
	}
	if stored["inputs"] == nil || stored["plan"] == nil {
		t.Fatal("stored plan must include inputs and plan for auditability")
	}
}

func TestPlanBudgetChangeReSolves(t *testing.T) {
	srv := newTestServer(t)
	do := func(budget float64) map[string]any {
		body := map[string]any{
			"budget": budget, "currency": "INR",
			"horizon": map[string]string{"from": "2026-12-01", "to": "2026-12-31"},
			"risk":    map[string]any{"beta": 0.0, "alpha": 0.95},
			"constraints": map[string]any{"reserve_frac": 0.0, "max_share_per_destination": 0, "max_share_per_supplier": 0, "min_fill_rate": 0, "allowed_structures": []string{}, "exploration_frac": 0},
			"segments": []string{},
		}
		b, _ := json.Marshal(body)
		req := httptest.NewRequest(http.MethodPost, "/api/v1/plan", bytes.NewReader(b))
		w := httptest.NewRecorder()
		srv.Routes().ServeHTTP(w, req)
		if w.Code != http.StatusOK {
			t.Fatalf("status %d body %s", w.Code, w.Body.String())
		}
		var resp map[string]any
		_ = json.Unmarshal(w.Body.Bytes(), &resp)
		return resp
	}
	r1 := do(5_000_000)
	r2 := do(15_000_000)
	if r1["cash_deployed"] == r2["cash_deployed"] {
		t.Fatalf("expected different cash_deployed for different budgets: %v vs %v", r1["cash_deployed"], r2["cash_deployed"])
	}
}
