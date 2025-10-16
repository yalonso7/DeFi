# election_fraud_detection_backend.py
"""
Blockchain DApp Backend for Election Fraud Detection
Integrates TensorFlow ML models, Web3 blockchain, and anomaly detection
"""

import tensorflow as tf
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify
from flask_cors import CORS
from web3 import Web3
from eth_account import Account
import json
from datetime import datetime
import hashlib
from scipy import stats
from sklearn.preprocessing import StandardScaler
import pickle

app = Flask(__name__)
CORS(app)

# ==================== BLOCKCHAIN CONFIGURATION ====================
class BlockchainManager:
    def __init__(self, provider_url='http://localhost:8545'):
        self.w3 = Web3(Web3.HTTPProvider(provider_url))
        self.contract_address = None
        self.contract_abi = None
        self.account = None
        
    def connect_contract(self, contract_address, abi_path):
        """Connect to deployed smart contract"""
        with open(abi_path, 'r') as f:
            self.contract_abi = json.load(f)
        self.contract_address = contract_address
        self.contract = self.w3.eth.contract(
            address=contract_address,
            abi=self.contract_abi
        )
        
    def record_vote(self, region_id, candidate_id, voter_hash):
        """Record vote on blockchain with privacy"""
        tx = self.contract.functions.castVote(
            region_id,
            candidate_id,
            voter_hash
        ).build_transaction({
            'from': self.account.address,
            'nonce': self.w3.eth.get_transaction_count(self.account.address),
            'gas': 200000,
            'gasPrice': self.w3.eth.gas_price
        })
        
        signed_tx = self.w3.eth.account.sign_transaction(tx, self.account.key)
        tx_hash = self.w3.eth.send_raw_transaction(signed_tx.rawTransaction)
        return self.w3.eth.wait_for_transaction_receipt(tx_hash)
    
    def verify_vote_integrity(self, block_number):
        """Verify blockchain integrity"""
        block = self.w3.eth.get_block(block_number)
        return {
            'hash': block['hash'].hex(),
            'parentHash': block['parentHash'].hex(),
            'timestamp': block['timestamp'],
            'transactions': len(block['transactions'])
        }


# ==================== ML FRAUD DETECTION MODEL ====================
class FraudDetectionML:
    def __init__(self, model_path=None):
        self.model = None
        self.scaler = StandardScaler()
        self.threshold = 0.65  # Fraud probability threshold
        
        if model_path:
            self.load_model(model_path)
        else:
            self.build_model()
    
    def build_model(self):
        """Build LSTM Neural Network for fraud detection"""
        self.model = tf.keras.Sequential([
            tf.keras.layers.Dense(128, activation='relu', input_shape=(10,)),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(32, activation='relu'),
            tf.keras.layers.Dense(16, activation='relu'),
            tf.keras.layers.Dense(1, activation='sigmoid')
        ])
        
        self.model.compile(
            optimizer='adam',
            loss='binary_crossentropy',
            metrics=['accuracy', 'precision', 'recall']
        )
    
    def extract_features(self, election_data):
        """Extract features from election data for ML model"""
        features = []
        
        for region in election_data:
            # Feature 1-2: Turnout metrics
            turnout_rate = region['actualVotes'] / max(region['registeredVoters'], 1)
            turnout_deviation = abs(turnout_rate - 0.70)  # Expected ~70% turnout
            
            # Feature 3-4: Vote distribution
            votes = [v['votes'] for v in region['votes']]
            vote_mean = np.mean(votes)
            vote_std = np.std(votes)
            vote_cv = vote_std / max(vote_mean, 1)  # Coefficient of variation
            
            # Feature 5: Benford's Law check (first digit distribution)
            benford_score = self.calculate_benford_score(votes)
            
            # Feature 6: Temporal anomaly (voting pattern over time)
            temporal_score = self.calculate_temporal_anomaly(region)
            
            # Feature 7-8: Statistical tests
            skewness = stats.skew(votes)
            kurtosis = stats.kurtosis(votes)
            
            # Feature 9: Invalid votes ratio
            invalid_ratio = region.get('invalidVotes', 0) / max(region['actualVotes'], 1)
            
            # Feature 10: Overvote detection
            overvote_flag = 1.0 if region['actualVotes'] > region['registeredVoters'] else 0.0
            
            features.append([
                turnout_rate,
                turnout_deviation,
                vote_cv,
                benford_score,
                temporal_score,
                skewness,
                kurtosis,
                invalid_ratio,
                overvote_flag,
                len(votes)
            ])
        
        return np.array(features)
    
    def calculate_benford_score(self, votes):
        """Apply Benford's Law to detect fabricated numbers"""
        if not votes:
            return 0.0
        
        first_digits = [int(str(abs(int(v)))[0]) for v in votes if v > 0]
        if not first_digits:
            return 0.0
        
        # Expected Benford distribution
        benford_expected = {i: np.log10(1 + 1/i) for i in range(1, 10)}
        
        # Observed distribution
        observed = {i: first_digits.count(i) / len(first_digits) for i in range(1, 10)}
        
        # Chi-square test
        chi_square = sum((observed.get(i, 0) - benford_expected[i])**2 / benford_expected[i] 
                        for i in range(1, 10))
        
        return min(chi_square / 10, 1.0)  # Normalize
    
    def calculate_temporal_anomaly(self, region):
        """Detect unusual voting patterns over time"""
        # Simulate temporal analysis (in production, use actual timestamps)
        if 'hourlyVotes' in region:
            hourly = region['hourlyVotes']
            # Check for sudden spikes
            mean_rate = np.mean(hourly)
            max_spike = max(hourly) / max(mean_rate, 1)
            return min(max_spike / 3, 1.0)
        return 0.0
    
    def predict_fraud(self, election_data):
        """Predict fraud probability for each region"""
        features = self.extract_features(election_data)
        features_scaled = self.scaler.fit_transform(features)
        
        predictions = self.model.predict(features_scaled)
        
        results = []
        for i, prob in enumerate(predictions):
            fraud_prob = float(prob[0])
            results.append({
                'region': election_data[i]['region'],
                'fraudProbability': round(fraud_prob * 100, 2),
                'classification': self.classify_risk(fraud_prob),
                'confidence': round(abs(fraud_prob - 0.5) * 2, 2),
                'features': {
                    'turnoutAnomaly': round(features[i][1], 3),
                    'benfordScore': round(features[i][3], 3),
                    'temporalAnomaly': round(features[i][4], 3),
                    'overvoteFlag': bool(features[i][8])
                }
            })
        
        return results
    
    def classify_risk(self, probability):
        """Classify fraud risk level"""
        if probability >= 0.70:
            return 'High Risk'
        elif probability >= 0.40:
            return 'Medium Risk'
        else:
            return 'Low Risk'
    
    def train_model(self, training_data, labels, epochs=50):
        """Train the fraud detection model"""
        features = self.extract_features(training_data)
        features_scaled = self.scaler.fit_transform(features)
        
        history = self.model.fit(
            features_scaled,
            labels,
            epochs=epochs,
            batch_size=32,
            validation_split=0.2,
            verbose=1
        )
        
        return history
    
    def save_model(self, model_path, scaler_path):
        """Save trained model and scaler"""
        self.model.save(model_path)
        with open(scaler_path, 'wb') as f:
            pickle.dump(self.scaler, f)
    
    def load_model(self, model_path, scaler_path=None):
        """Load pre-trained model"""
        self.model = tf.keras.models.load_model(model_path)
        if scaler_path:
            with open(scaler_path, 'rb') as f:
                self.scaler = pickle.load(f)


# ==================== ANOMALY DETECTION ENGINE ====================
class AnomalyDetector:
    def __init__(self):
        self.anomaly_types = {
            'TURNOUT_SPIKE': {'threshold': 0.95, 'severity': 'high'},
            'OVERVOTE': {'threshold': 1.0, 'severity': 'critical'},
            'BENFORD_VIOLATION': {'threshold': 0.7, 'severity': 'medium'},
            'TIMESTAMP_MANIPULATION': {'threshold': 0.8, 'severity': 'high'},
            'DUPLICATE_VOTES': {'threshold': 0.01, 'severity': 'critical'},
            'INVALID_RATIO': {'threshold': 0.15, 'severity': 'medium'}
        }
    
    def detect_all_anomalies(self, election_data):
        """Run all anomaly detection algorithms"""
        anomalies = []
        
        for region in election_data:
            # Check turnout spike
            turnout = region['actualVotes'] / region['registeredVoters']
            if turnout > self.anomaly_types['TURNOUT_SPIKE']['threshold']:
                anomalies.append({
                    'type': 'Suspicious Turnout',
                    'severity': 'high',
                    'region': region['region'],
                    'description': f"Turnout rate of {turnout*100:.1f}% exceeds normal statistical bounds",
                    'confidence': min((turnout - 0.95) * 10, 1.0),
                    'timestamp': datetime.now().isoformat()
                })
            
            # Check overvoting
            if region['actualVotes'] > region['registeredVoters']:
                anomalies.append({
                    'type': 'Vote Count Mismatch',
                    'severity': 'critical',
                    'region': region['region'],
                    'description': f"Actual votes ({region['actualVotes']}) exceed registered voters ({region['registeredVoters']})",
                    'confidence': 0.98,
                    'timestamp': datetime.now().isoformat()
                })
            
            # Check vote distribution using chi-square test
            votes = [v['votes'] for v in region['votes']]
            if len(votes) > 1:
                chi_stat, p_value = stats.chisquare(votes)
                if p_value < 0.05:  # Significant deviation
                    anomalies.append({
                        'type': 'Irregular Distribution',
                        'severity': 'medium',
                        'region': region['region'],
                        'description': f"Vote distribution shows statistical irregularity (p={p_value:.4f})",
                        'confidence': 1 - p_value,
                        'timestamp': datetime.now().isoformat()
                    })
            
            # Check for duplicate voter IDs (if available)
            if 'voterHashes' in region:
                unique_voters = len(set(region['voterHashes']))
                total_votes = len(region['voterHashes'])
                duplicate_ratio = (total_votes - unique_voters) / total_votes
                
                if duplicate_ratio > 0.01:
                    anomalies.append({
                        'type': 'Duplicate Votes Detected',
                        'severity': 'critical',
                        'region': region['region'],
                        'description': f"{duplicate_ratio*100:.2f}% of votes are duplicates",
                        'confidence': 0.95,
                        'timestamp': datetime.now().isoformat()
                    })
        
        return anomalies
    
    def generate_anomaly_report(self, anomalies):
        """Generate detailed anomaly report"""
        severity_counts = {
            'critical': len([a for a in anomalies if a['severity'] == 'critical']),
            'high': len([a for a in anomalies if a['severity'] == 'high']),
            'medium': len([a for a in anomalies if a['severity'] == 'medium']),
            'low': len([a for a in anomalies if a['severity'] == 'low'])
        }
        
        return {
            'totalAnomalies': len(anomalies),
            'severityBreakdown': severity_counts,
            'riskScore': self.calculate_risk_score(severity_counts),
            'anomalies': anomalies,
            'generatedAt': datetime.now().isoformat()
        }
    
    def calculate_risk_score(self, severity_counts):
        """Calculate overall election integrity risk score"""
        weights = {'critical': 10, 'high': 5, 'medium': 2, 'low': 1}
        weighted_sum = sum(severity_counts[s] * weights[s] for s in severity_counts)
        return min(weighted_sum / 20, 1.0)


# ==================== API ENDPOINTS ====================
blockchain_manager = BlockchainManager()
fraud_detector = FraudDetectionML()
anomaly_detector = AnomalyDetector()

@app.route('/api/analyze-election', methods=['POST'])
def analyze_election():
    """Main endpoint for election analysis"""
    data = request.json
    election_data = data.get('regions', [])
    
    # Run ML fraud detection
    ml_predictions = fraud_detector.predict_fraud(election_data)
    
    # Run anomaly detection
    anomalies = anomaly_detector.detect_all_anomalies(election_data)
    anomaly_report = anomaly_detector.generate_anomaly_report(anomalies)
    
    return jsonify({
        'status': 'success',
        'mlPredictions': ml_predictions,
        'anomalyReport': anomaly_report,
        'timestamp': datetime.now().isoformat()
    })

@app.route('/api/record-vote', methods=['POST'])
def record_vote():
    """Record vote on blockchain"""
    data = request.json
    try:
        receipt = blockchain_manager.record_vote(
            data['regionId'],
            data['candidateId'],
            data['voterHash']
        )
        return jsonify({
            'status': 'success',
            'transactionHash': receipt['transactionHash'].hex(),
            'blockNumber': receipt['blockNumber']
        })
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/api/verify-blockchain', methods=['GET'])
def verify_blockchain():
    """Verify blockchain integrity"""
    block_number = request.args.get('block', 'latest')
    block_info = blockchain_manager.verify_vote_integrity(block_number)
    return jsonify(block_info)

@app.route('/api/train-model', methods=['POST'])
def train_model():
    """Train ML model with new data"""
    data = request.json
    training_data = data.get('trainingData', [])
    labels = data.get('labels', [])
    
    history = fraud_detector.train_model(training_data, labels)
    
    return jsonify({
        'status': 'success',
        'finalAccuracy': float(history.history['accuracy'][-1]),
        'finalLoss': float(history.history['loss'][-1])
    })

@app.route('/api/benford-analysis', methods=['POST'])
def benford_analysis():
    """Perform Benford's Law analysis on vote counts"""
    data = request.json
    votes = data.get('votes', [])
    
    score = fraud_detector.calculate_benford_score(votes)
    
    return jsonify({
        'benfordScore': score,
        'interpretation': 'High suspicion' if score > 0.7 else 'Normal pattern',
        'isAnomaly': score > 0.7
    })

@app.route('/api/generate-report', methods=['POST'])
def generate_report():
    """Generate comprehensive election integrity report"""
    data = request.json
    election_data = data.get('regions', [])
    
    ml_predictions = fraud_detector.predict_fraud(election_data)
    anomalies = anomaly_detector.detect_all_anomalies(election_data)
    anomaly_report = anomaly_detector.generate_anomaly_report(anomalies)
    
    # Calculate summary statistics
    high_risk_regions = [p for p in ml_predictions if p['classification'] == 'High Risk']
    total_votes = sum(r['actualVotes'] for r in election_data)
    
    report = {
        'executiveSummary': {
            'totalRegions': len(election_data),
            'totalVotes': total_votes,
            'highRiskRegions': len(high_risk_regions),
            'criticalAnomalies': anomaly_report['severityBreakdown']['critical'],
            'overallRiskScore': anomaly_report['riskScore'],
            'recommendation': 'Manual audit required' if anomaly_report['riskScore'] > 0.6 else 'Acceptable'
        },
        'mlAnalysis': ml_predictions,
        'anomalyReport': anomaly_report,
        'generatedAt': datetime.now().isoformat()
    }
    
    return jsonify(report)

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({'status': 'healthy', 'timestamp': datetime.now().isoformat()})


if __name__ == '__main__':
    print("🚀 Election Fraud Detection Backend Starting...")
    print("📊 ML Model: TensorFlow Neural Network")
    print("⛓️  Blockchain: Web3 Integration Ready")
    print("🔍 Anomaly Detection: Active")
    app.run(host='0.0.0.0', port=5000, debug=True)
