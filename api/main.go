package main

// TBO Inventory Optimization API (Go port of water-filling allocator).
// Assumption: default dataset at data/api_demo, store at data/api_plans; simulator outputs under stated assumptions.

import (
	"log"
	"net/http"
	"os"
	"path/filepath"

	"tbo-inventory-api/internal/api"
	"tbo-inventory-api/internal/store"
)

func main() {
	port := os.Getenv("PORT")
	if port == "" {
		port = "8080"
	}
	root := os.Getenv("TBO_ROOT")
	if root == "" {
		if wd, err := os.Getwd(); err == nil {
			root = filepath.Dir(wd)
			if filepath.Base(wd) != "api" {
				root = wd
			}
		} else {
			root = "."
		}
	}
	dataDir := os.Getenv("TBO_DATA_DIR")
	if dataDir == "" {
		dataDir = filepath.Join(root, "data", "api_demo")
	}
	storeDir := os.Getenv("TBO_STORE_DIR")
	if storeDir == "" {
		storeDir = filepath.Join(root, "data", "api_plans")
	}
	st, err := store.New(storeDir)
	if err != nil {
		log.Fatalf("store init: %v", err)
	}
	srv, err := api.NewServer(root, dataDir, st)
	if err != nil {
		log.Fatalf("server init: %v", err)
	}
	log.Printf("TBO API listening on 0.0.0.0:%s root=%s data=%s", port, root, dataDir)
	log.Fatal(http.ListenAndServe("0.0.0.0:"+port, srv.Routes()))
}
