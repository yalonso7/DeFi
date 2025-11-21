package main

import (
	"encoding/json"
	"fmt"
	"log"
	"math"
	"os"
	"os/signal"
	"sort"
	"sync"
	"syscall"
	"time"

	"github.com/gorilla/websocket"
)

// TransactionMetrics represents transaction data
type TransactionMetrics struct {
	Timestamp   time.Time `json:"timestamp"`
	TxHash      string    `json:"tx_hash"`
	TxType      string    `json:"tx_type"`
	Amount      float64   `json:"amount"`
	Fee         float64   `json:"fee"`
	Account     string    `json:"account"`
	Destination string    `json:"destination"`
	Success     bool      `json:"success"`
	LedgerIndex int64     `json:"ledger_index"`
}

// AnomalyReport represents an anomaly detection result
type AnomalyReport struct {
	Timestamp    time.Time              `json:"timestamp"`
	TxHash       string                 `json:"tx_hash"`
	AnomalyScore float64                `json:"anomaly_score"`
	IsAnomaly    bool                   `json:"is_anomaly"`
	Reason       string                 `json:"reason"`
	Metrics      map[string]interface{} `json:"metrics"`
}

// Statistics tracks monitoring statistics
type Statistics struct {
	TotalTransactions int64              `json:"total_transactions"`
	TotalAnomalies    int64              `json:"total_anomalies"`
	TxTypes           map[string]int64   `json:"tx_types"`
	AvgAmount         float64            `json:"avg_amount"`
	AvgFee            float64            `json:"avg_fee"`
	TotalVolume       float64            `json:"total_volume"`
}

// XRPLMonitor is the main monitoring structure
type XRPLMonitor struct {
	websocketURL      string
	windowSize        int
	anomalyThreshold  float64
	
	// Data storage
	transactionBuffer []*TransactionMetrics
	metricsHistory    []*TransactionMetrics
	anomalies         []*AnomalyReport
	
	// Statistics
	stats *Statistics
	
	// Synchronization
	mu sync.RWMutex
	
	// Feature statistics for normalization
	featureMeans []float64
	featureStds  []float64
	isTrained    bool
}

// NewXRPLMonitor creates a new monitor instance
func NewXRPLMonitor(websocketURL string, windowSize int, anomalyThreshold float64) *XRPLMonitor {
	return &XRPLMonitor{
		websocketURL:     websocketURL,
		windowSize:       windowSize,
		anomalyThreshold: anomalyThreshold,
		stats: &Statistics{
			TxTypes: make(map[string]int64),
		},
		transactionBuffer: make([]*TransactionMetrics, 0, windowSize),
		metricsHistory:    make([]*TransactionMetrics, 0),
		anomalies:         make([]*AnomalyReport, 0),
	}
}

// extractFeatures converts transaction metrics to feature vector
func (m *XRPLMonitor) extractFeatures(metrics *TransactionMetrics) []float64 {
	successVal := 0.0
	if metrics.Success {
		successVal = 1.0
	}
	
	destLen := 0.0
	if metrics.Destination != "" {
		destLen = float64(len(metrics.Destination))
	}
	
	return []float64{
		metrics.Amount,
		metrics.Fee,
		successVal,
		float64(metrics.LedgerIndex),
		float64(len(metrics.Account)),
		destLen,
	}
}

// normalizeFeatures normalizes feature vector
func (m *XRPLMonitor) normalizeFeatures(features []float64) []float64 {
	if m.featureMeans == nil {
		return features
	}
	
	normalized := make([]float64, len(features))
	for i := range features {
		normalized[i] = (features[i] - m.featureMeans[i]) / (m.featureStds[i] + 1e-8)
	}
	return normalized
}

// trainDetector trains the anomaly detection model
func (m *XRPLMonitor) trainDetector() {
	m.mu.Lock()
	defer m.mu.Unlock()
	
	if len(m.transactionBuffer) < 20 {
		log.Println("Insufficient data for training")
		return
	}
	
	// Extract features from all transactions in buffer
	allFeatures := make([][]float64, len(m.transactionBuffer))
	for i, tx := range m.transactionBuffer {
		allFeatures[i] = m.extractFeatures(tx)
	}
	
	// Calculate mean and std for each feature
	numFeatures := len(allFeatures[0])
	m.featureMeans = make([]float64, numFeatures)
	m.featureStds = make([]float64, numFeatures)
	
	// Calculate means
	for _, features := range allFeatures {
		for j, val := range features {
			m.featureMeans[j] += val
		}
	}
	for j := range m.featureMeans {
		m.featureMeans[j] /= float64(len(allFeatures))
	}
	
	// Calculate standard deviations
	for _, features := range allFeatures {
		for j, val := range features {
			diff := val - m.featureMeans[j]
			m.featureStds[j] += diff * diff
		}
	}
	for j := range m.featureStds {
		m.featureStds[j] = math.Sqrt(m.featureStds[j] / float64(len(allFeatures)))
	}
	
	m.isTrained = true
	log.Println("Anomaly detector trained successfully")
}

// detectAnomaly detects if a transaction is anomalous using statistical methods
func (m *XRPLMonitor) detectAnomaly(metrics *TransactionMetrics) (float64, bool, string) {
	if !m.isTrained {
		return 0.0, false, "Detector not trained"
	}
	
	features := m.extractFeatures(metrics)
	normalized := m.normalizeFeatures(features)
	
	// Calculate Mahalanobis-like distance (simplified)
	anomalyScore := 0.0
	for _, val := range normalized {
		anomalyScore += val * val
	}
	anomalyScore = math.Sqrt(anomalyScore)
	
	isAnomaly := anomalyScore > m.anomalyThreshold
	reason := "Normal transaction"
	
	if isAnomaly {
		m.mu.RLock()
		avgAmount := m.stats.AvgAmount
		avgFee := m.stats.AvgFee
		m.mu.RUnlock()
		
		if metrics.Amount > avgAmount*10 {
			reason = "Unusually large transaction amount"
		} else if metrics.Fee > avgFee*5 {
			reason = "Unusually high transaction fee"
		} else {
			reason = "Unusual transaction pattern detected"
		}
	}
	
	return anomalyScore, isAnomaly, reason
}

// updateStatistics updates running statistics
func (m *XRPLMonitor) updateStatistics(metrics *TransactionMetrics) {
	m.mu.Lock()
	defer m.mu.Unlock()
	
	m.stats.TotalTransactions++
	m.stats.TxTypes[metrics.TxType]++
	
	// Update running averages
	n := float64(m.stats.TotalTransactions)
	m.stats.AvgAmount = (m.stats.AvgAmount*(n-1) + metrics.Amount) / n
	m.stats.AvgFee = (m.stats.AvgFee*(n-1) + metrics.Fee) / n
	m.stats.TotalVolume += metrics.Amount
}

// processTransaction processes incoming transaction data
func (m *XRPLMonitor) processTransaction(txData map[string]interface{}) {
	tx, ok := txData["transaction"].(map[string]interface{})
	if !ok {
		return
	}
	
	meta, _ := txData["meta"].(map[string]interface{})
	
	// Parse amount
	amount := 0.0
	if amountVal, ok := tx["Amount"]; ok {
		switch v := amountVal.(type) {
		case string:
			var amt float64
			fmt.Sscanf(v, "%f", &amt)
			amount = amt / 1_000_000.0 // Convert drops to XRP
		case map[string]interface{}:
			if val, ok := v["value"].(string); ok {
				fmt.Sscanf(val, "%f", &amount)
			}
		}
	}
	
	// Parse fee
	fee := 0.0
	if feeVal, ok := tx["Fee"].(string); ok {
		var f float64
		fmt.Sscanf(feeVal, "%f", &f)
		fee = f / 1_000_000.0
	}
	
	// Get ledger index
	ledgerIndex := int64(0)
	if idx, ok := txData["ledger_index"].(float64); ok {
		ledgerIndex = int64(idx)
	}
	
	// Create metrics
	metrics := &TransactionMetrics{
		Timestamp:   time.Now(),
		TxHash:      getString(tx, "hash"),
		TxType:      getString(tx, "TransactionType"),
		Amount:      amount,
		Fee:         fee,
		Account:     getString(tx, "Account"),
		Destination: getString(tx, "Destination"),
		Success:     getString(meta, "TransactionResult") == "tesSUCCESS",
		LedgerIndex: ledgerIndex,
	}
	
	// Add to buffer
	m.mu.Lock()
	if len(m.transactionBuffer) >= m.windowSize {
		m.transactionBuffer = m.transactionBuffer[1:]
	}
	m.transactionBuffer = append(m.transactionBuffer, metrics)
	m.metricsHistory = append(m.metricsHistory, metrics)
	bufferFull := len(m.transactionBuffer) == m.windowSize
	m.mu.Unlock()
	
	// Update statistics
	m.updateStatistics(metrics)
	
	// Detect anomalies
	if m.isTrained {
		score, isAnomaly, reason := m.detectAnomaly(metrics)
		
		if isAnomaly {
			m.mu.Lock()
			m.stats.TotalAnomalies++
			
			report := &AnomalyReport{
				Timestamp:    metrics.Timestamp,
				TxHash:       metrics.TxHash,
				AnomalyScore: score,
				IsAnomaly:    isAnomaly,
				Reason:       reason,
				Metrics:      structToMap(metrics),
			}
			m.anomalies = append(m.anomalies, report)
			m.mu.Unlock()
			
			log.Printf("⚠️  ANOMALY DETECTED: %s (Score: %.2f)", reason, score)
			log.Printf("   Transaction: %s", metrics.TxHash)
		}
	}
	
	// Log transaction
	log.Printf("TX: %s | Amount: %.2f XRP | Fee: %.6f XRP | Success: %v",
		metrics.TxType, metrics.Amount, metrics.Fee, metrics.Success)
	
	// Train detector when buffer is full
	if bufferFull && !m.isTrained {
		log.Println("Buffer full, training anomaly detector...")
		m.trainDetector()
	}
}

// startMonitoring starts the WebSocket connection and monitoring
func (m *XRPLMonitor) startMonitoring() error {
	log.Printf("Connecting to XRPL at %s", m.websocketURL)
	
	conn, _, err := websocket.DefaultDialer.Dial(m.websocketURL, nil)
	if err != nil {
		return fmt.Errorf("dial error: %w", err)
	}
	defer conn.Close()
	
	// Subscribe to transaction stream
	subscribeMsg := map[string]interface{}{
		"command": "subscribe",
		"streams": []string{"transactions"},
	}
	
	if err := conn.WriteJSON(subscribeMsg); err != nil {
		return fmt.Errorf("subscribe error: %w", err)
	}
	
	log.Println("✓ Subscribed to transaction stream")
	
	// Read messages
	for {
		var msg map[string]interface{}
		if err := conn.ReadJSON(&msg); err != nil {
			return fmt.Errorf("read error: %w", err)
		}
		
		if msgType, ok := msg["type"].(string); ok && msgType == "transaction" {
			m.processTransaction(msg)
		}
	}
}

// generateAnalyticsReport generates a comprehensive report
func (m *XRPLMonitor) generateAnalyticsReport() map[string]interface{} {
	m.mu.RLock()
	defer m.mu.RUnlock()
	
	report := map[string]interface{}{
		"report_timestamp":     time.Now().Format(time.RFC3339),
		"monitoring_duration":  len(m.metricsHistory),
		"statistics":           m.stats,
		"recent_anomalies":     make([]map[string]interface{}, 0),
		"top_transaction_types": make([]map[string]interface{}, 0),
		"performance_metrics":  make(map[string]interface{}),
	}
	
	// Recent anomalies (last 10)
	startIdx := 0
	if len(m.anomalies) > 10 {
		startIdx = len(m.anomalies) - 10
	}
	recentAnomalies := make([]map[string]interface{}, 0)
	for _, a := range m.anomalies[startIdx:] {
		recentAnomalies = append(recentAnomalies, map[string]interface{}{
			"timestamp": a.Timestamp.Format(time.RFC3339),
			"tx_hash":   a.TxHash,
			"score":     math.Round(a.AnomalyScore*100) / 100,
			"reason":    a.Reason,
		})
	}
	report["recent_anomalies"] = recentAnomalies
	
	// Top transaction types
	type txTypePair struct {
		Type  string
		Count int64
	}
	pairs := make([]txTypePair, 0)
	for k, v := range m.stats.TxTypes {
		pairs = append(pairs, txTypePair{k, v})
	}
	sort.Slice(pairs, func(i, j int) bool {
		return pairs[i].Count > pairs[j].Count
	})
	
	topTypes := make([]map[string]interface{}, 0)
	for i := 0; i < len(pairs) && i < 5; i++ {
		topTypes = append(topTypes, map[string]interface{}{
			"type":  pairs[i].Type,
			"count": pairs[i].Count,
		})
	}
	report["top_transaction_types"] = topTypes
	
	// Performance metrics
	if len(m.metricsHistory) > 0 {
		amounts := make([]float64, 0)
		for _, metrics := range m.metricsHistory {
			if metrics.Amount > 0 {
				amounts = append(amounts, metrics.Amount)
			}
		}
		
		if len(amounts) > 0 {
			sort.Float64s(amounts)
			median := amounts[len(amounts)/2]
			max := amounts[len(amounts)-1]
			
			anomalyRate := 0.0
			if m.stats.TotalTransactions > 0 {
				anomalyRate = float64(m.stats.TotalAnomalies) / float64(m.stats.TotalTransactions) * 100
			}
			
			report["performance_metrics"] = map[string]interface{}{
				"total_volume_xrp":       math.Round(m.stats.TotalVolume*100) / 100,
				"median_amount_xrp":      math.Round(median*100) / 100,
				"max_amount_xrp":         math.Round(max*100) / 100,
				"anomaly_rate_percent":   math.Round(anomalyRate*100) / 100,
			}
		}
	}
	
	return report
}

// saveReport saves the analytics report to a file
func (m *XRPLMonitor) saveReport(filename string) error {
	if filename == "" {
		filename = fmt.Sprintf("xrpl_report_%s.json", time.Now().Format("20060102_150405"))
	}
	
	report := m.generateAnalyticsReport()
	data, err := json.MarshalIndent(report, "", "  ")
	if err != nil {
		return err
	}
	
	if err := os.WriteFile(filename, data, 0644); err != nil {
		return err
	}
	
	log.Printf("Report saved to %s", filename)
	return nil
}

// Helper functions
func getString(m map[string]interface{}, key string) string {
	if val, ok := m[key].(string); ok {
		return val
	}
	return ""
}

func structToMap(v interface{}) map[string]interface{} {
	data, _ := json.Marshal(v)
	var result map[string]interface{}
	json.Unmarshal(data, &result)
	return result
}

func main() {
	monitor := NewXRPLMonitor(
		"wss://xrplcluster.com/",
		100,
		2.5,
	)
	
	// Handle graceful shutdown
	sigChan := make(chan os.Signal, 1)
	signal.Notify(sigChan, os.Interrupt, syscall.SIGTERM)
	
	go func() {
		<-sigChan
		log.Println("\nMonitoring stopped by user")
		
		// Generate final report
		report := monitor.generateAnalyticsReport()
		fmt.Println("\n" + strings.Repeat("=", 60))
		fmt.Println("FINAL ANALYTICS REPORT")
		fmt.Println(strings.Repeat("=", 60))
		
		reportJSON, _ := json.MarshalIndent(report, "", "  ")
		fmt.Println(string(reportJSON))
		
		// Save report
		monitor.saveReport("")
		
		os.Exit(0)
	}()
	
	// Start monitoring
	if err := monitor.startMonitoring(); err != nil {
		log.Fatalf("Monitoring error: %v", err)
	}
}


