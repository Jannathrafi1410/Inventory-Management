package store

// File-based plan store with versioned inputs for auditability.
//
// Assumption: plans persisted as JSON files under store dir (SQLite alternative); each plan records
// full request inputs and response for reproducibility. Simulator outputs under stated assumptions.

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"sync"
	"time"
)

type StoredPlan struct {
	PlanID    string          `json:"plan_id"`
	Version   int             `json:"version"`
	CreatedAt string          `json:"created_at"`
	Inputs    json.RawMessage `json:"inputs"`
	Plan      json.RawMessage `json:"plan"`
}

type Store struct {
	dir string
	mu  sync.RWMutex
}

func New(dir string) (*Store, error) {
	if err := os.MkdirAll(dir, 0o755); err != nil {
		return nil, err
	}
	return &Store{dir: dir}, nil
}

func (s *Store) Save(planID string, inputs, plan any) (*StoredPlan, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	inb, err := json.Marshal(inputs)
	if err != nil {
		return nil, err
	}
	pb, err := json.Marshal(plan)
	if err != nil {
		return nil, err
	}
	// determine version
	version := 1
	pattern := filepath.Join(s.dir, fmt.Sprintf("%s.v*.json", planID))
	if matches, _ := filepath.Glob(pattern); len(matches) > 0 {
		version = len(matches) + 1
	}
	sp := &StoredPlan{PlanID: planID, Version: version, CreatedAt: time.Now().UTC().Format(time.RFC3339), Inputs: inb, Plan: pb}
	b, err := json.MarshalIndent(sp, "", "  ")
	if err != nil {
		return nil, err
	}
	name := filepath.Join(s.dir, fmt.Sprintf("%s.v%d.json", planID, version))
	if err := os.WriteFile(name, b, 0o644); err != nil {
		return nil, err
	}
	// also write/update latest pointer
	latest := filepath.Join(s.dir, fmt.Sprintf("%s.json", planID))
	if err := os.WriteFile(latest, b, 0o644); err != nil {
		return nil, err
	}
	return sp, nil
}

func (s *Store) Get(planID string) (*StoredPlan, error) {
	s.mu.RLock()
	defer s.mu.RUnlock()
	latest := filepath.Join(s.dir, fmt.Sprintf("%s.json", planID))
	b, err := os.ReadFile(latest)
	if err != nil {
		return nil, err
	}
	var sp StoredPlan
	if err := json.Unmarshal(b, &sp); err != nil {
		return nil, err
	}
	return &sp, nil
}
