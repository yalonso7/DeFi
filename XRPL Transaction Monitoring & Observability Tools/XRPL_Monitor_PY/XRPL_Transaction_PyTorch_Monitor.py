"""
XRPL Transaction Monitor with PyTorch Anomaly Detection
Monitors XRP Ledger transactions, detects anomalies, and generates analytics reports
"""

import asyncio
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from collections import deque
from dataclasses import dataclass, asdict
import statistics

import torch
import torch.nn as nn
import numpy as np
from xrpl.asyncio.clients import AsyncWebsocketClient
from xrpl.models import Subscribe, StreamParameter

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class TransactionMetrics:
    """Metrics for XRPL transactions"""
    timestamp: datetime
    tx_hash: str
    tx_type: str
    amount: float
    fee: float
    account: str
    destination: Optional[str]
    success: bool
    ledger_index: int


@dataclass
class AnomalyReport:
    """Anomaly detection report"""
    timestamp: datetime
    tx_hash: str
    anomaly_score: float
    is_anomaly: bool
    reason: str
    metrics: Dict


class AutoencoderAnomalyDetector(nn.Module):
    """Autoencoder neural network for anomaly detection"""
    
    def __init__(self, input_dim: int, hidden_dims: List[int] = [16, 8, 4]):
        super().__init__()
        
        # Encoder
        encoder_layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            encoder_layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.BatchNorm1d(hidden_dim)
            ])
            prev_dim = hidden_dim
        self.encoder = nn.Sequential(*encoder_layers)
        
        # Decoder
        decoder_layers = []
        for i in range(len(hidden_dims) - 1, -1, -1):
            next_dim = hidden_dims[i - 1] if i > 0 else input_dim
            decoder_layers.extend([
                nn.Linear(hidden_dims[i], next_dim),
                nn.ReLU() if i > 0 else nn.Identity()
            ])
        self.decoder = nn.Sequential(*decoder_layers)
    
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


class XRPLMonitor:
    """Main monitoring class for XRPL transactions"""
    
    def __init__(self, websocket_url: str = "wss://xrplcluster.com/",
                 window_size: int = 100, anomaly_threshold: float = 2.5):
        self.websocket_url = websocket_url
        self.window_size = window_size
        self.anomaly_threshold = anomaly_threshold
        
        # Data storage
        self.transaction_buffer = deque(maxlen=window_size)
        self.metrics_history: List[TransactionMetrics] = []
        self.anomalies: List[AnomalyReport] = []
        
        # Statistics tracking
        self.stats = {
            'total_transactions': 0,
            'total_anomalies': 0,
            'tx_types': {},
            'avg_amount': 0.0,
            'avg_fee': 0.0
        }
        
        # Anomaly detector
        self.detector = AutoencoderAnomalyDetector(input_dim=6)
        self.detector.eval()
        self.is_trained = False
        
        # Feature scaling parameters
        self.feature_means = None
        self.feature_stds = None
    
    def extract_features(self, metrics: TransactionMetrics) -> np.ndarray:
        """Extract numerical features from transaction metrics"""
        features = [
            metrics.amount,
            metrics.fee,
            float(metrics.success),
            metrics.ledger_index,
            len(metrics.account),
            len(metrics.destination) if metrics.destination else 0
        ]
        return np.array(features, dtype=np.float32)
    
    def normalize_features(self, features: np.ndarray) -> np.ndarray:
        """Normalize features using stored mean and std"""
        if self.feature_means is None:
            return features
        return (features - self.feature_means) / (self.feature_stds + 1e-8)
    
    def train_detector(self, epochs: int = 50):
        """Train the anomaly detector on collected data"""
        if len(self.transaction_buffer) < 20:
            logger.warning("Insufficient data for training")
            return
        
        # Prepare training data
        features_list = [self.extract_features(m) for m in self.transaction_buffer]
        X = np.array(features_list)
        
        # Calculate normalization parameters
        self.feature_means = X.mean(axis=0)
        self.feature_stds = X.std(axis=0)
        
        # Normalize
        X_norm = self.normalize_features(X)
        X_tensor = torch.FloatTensor(X_norm)
        
        # Training
        optimizer = torch.optim.Adam(self.detector.parameters(), lr=0.001)
        criterion = nn.MSELoss()
        
        self.detector.train()
        for epoch in range(epochs):
            optimizer.zero_grad()
            outputs = self.detector(X_tensor)
            loss = criterion(outputs, X_tensor)
            loss.backward()
            optimizer.step()
            
            if (epoch + 1) % 10 == 0:
                logger.info(f"Training epoch {epoch+1}/{epochs}, Loss: {loss.item():.6f}")
        
        self.detector.eval()
        self.is_trained = True
        logger.info("Anomaly detector trained successfully")
    
    def detect_anomaly(self, metrics: TransactionMetrics) -> Tuple[float, bool, str]:
        """Detect if a transaction is anomalous"""
        if not self.is_trained:
            return 0.0, False, "Detector not trained"
        
        # Extract and normalize features
        features = self.extract_features(metrics)
        features_norm = self.normalize_features(features)
        features_tensor = torch.FloatTensor(features_norm).unsqueeze(0)
        
        # Get reconstruction
        with torch.no_grad():
            reconstruction = self.detector(features_tensor)
            mse = nn.functional.mse_loss(reconstruction, features_tensor).item()
        
        # Calculate anomaly score
        anomaly_score = mse * 100  # Scale for readability
        is_anomaly = anomaly_score > self.anomaly_threshold
        
        # Determine reason
        reason = "Normal transaction"
        if is_anomaly:
            if metrics.amount > self.stats['avg_amount'] * 10:
                reason = "Unusually large transaction amount"
            elif metrics.fee > self.stats['avg_fee'] * 5:
                reason = "Unusually high transaction fee"
            else:
                reason = "Unusual transaction pattern detected"
        
        return anomaly_score, is_anomaly, reason
    
    def update_statistics(self, metrics: TransactionMetrics):
        """Update running statistics"""
        self.stats['total_transactions'] += 1
        
        # Track transaction types
        tx_type = metrics.tx_type
        self.stats['tx_types'][tx_type] = self.stats['tx_types'].get(tx_type, 0) + 1
        
        # Update averages (running average)
        n = self.stats['total_transactions']
        self.stats['avg_amount'] = (
            (self.stats['avg_amount'] * (n - 1) + metrics.amount) / n
        )
        self.stats['avg_fee'] = (
            (self.stats['avg_fee'] * (n - 1) + metrics.fee) / n
        )
    
    def process_transaction(self, tx_data: Dict):
        """Process incoming transaction data"""
        try:
            # Extract transaction details
            tx = tx_data.get('transaction', {})
            meta = tx_data.get('meta', {})
            
            # Parse amount
            amount = 0.0
            if 'Amount' in tx:
                if isinstance(tx['Amount'], str):
                    amount = float(tx['Amount']) / 1_000_000  # Convert drops to XRP
                elif isinstance(tx['Amount'], dict):
                    amount = float(tx['Amount'].get('value', 0))
            
            # Create metrics object
            metrics = TransactionMetrics(
                timestamp=datetime.now(),
                tx_hash=tx.get('hash', ''),
                tx_type=tx.get('TransactionType', 'Unknown'),
                amount=amount,
                fee=float(tx.get('Fee', 0)) / 1_000_000,
                account=tx.get('Account', ''),
                destination=tx.get('Destination'),
                success=meta.get('TransactionResult') == 'tesSUCCESS',
                ledger_index=tx_data.get('ledger_index', 0)
            )
            
            # Add to buffer and history
            self.transaction_buffer.append(metrics)
            self.metrics_history.append(metrics)
            
            # Update statistics
            self.update_statistics(metrics)
            
            # Detect anomalies
            if self.is_trained:
                score, is_anomaly, reason = self.detect_anomaly(metrics)
                
                if is_anomaly:
                    self.stats['total_anomalies'] += 1
                    anomaly_report = AnomalyReport(
                        timestamp=metrics.timestamp,
                        tx_hash=metrics.tx_hash,
                        anomaly_score=score,
                        is_anomaly=is_anomaly,
                        reason=reason,
                        metrics=asdict(metrics)
                    )
                    self.anomalies.append(anomaly_report)
                    logger.warning(f"ANOMALY DETECTED: {reason} (Score: {score:.2f})")
                    logger.warning(f"Transaction: {metrics.tx_hash}")
            
            # Log transaction
            logger.info(
                f"TX: {metrics.tx_type} | Amount: {metrics.amount:.2f} XRP | "
                f"Fee: {metrics.fee:.6f} XRP | Success: {metrics.success}"
            )
            
            # Train detector periodically
            if len(self.transaction_buffer) == self.window_size and not self.is_trained:
                logger.info("Buffer full, training anomaly detector...")
                self.train_detector()
            
        except Exception as e:
            logger.error(f"Error processing transaction: {e}")
    
    async def start_monitoring(self):
        """Start monitoring XRPL transactions"""
        logger.info(f"Connecting to XRPL at {self.websocket_url}")
        
        async with AsyncWebsocketClient(self.websocket_url) as client:
            # Subscribe to transaction stream
            subscribe_request = Subscribe(
                streams=[StreamParameter.TRANSACTIONS]
            )
            
            await client.send(subscribe_request)
            logger.info("Subscribed to transaction stream")
            
            # Process incoming messages
            async for message in client:
                if isinstance(message, dict) and message.get('type') == 'transaction':
                    self.process_transaction(message)
    
    def generate_analytics_report(self) -> Dict:
        """Generate comprehensive analytics report"""
        report = {
            'report_timestamp': datetime.now().isoformat(),
            'monitoring_duration': len(self.metrics_history),
            'statistics': self.stats.copy(),
            'recent_anomalies': [],
            'top_transaction_types': [],
            'performance_metrics': {}
        }
        
        # Recent anomalies (last 10)
        recent_anomalies = sorted(
            self.anomalies[-10:],
            key=lambda x: x.timestamp,
            reverse=True
        )
        report['recent_anomalies'] = [
            {
                'timestamp': a.timestamp.isoformat(),
                'tx_hash': a.tx_hash,
                'score': round(a.anomaly_score, 2),
                'reason': a.reason
            }
            for a in recent_anomalies
        ]
        
        # Top transaction types
        sorted_types = sorted(
            self.stats['tx_types'].items(),
            key=lambda x: x[1],
            reverse=True
        )
        report['top_transaction_types'] = [
            {'type': k, 'count': v} for k, v in sorted_types[:5]
        ]
        
        # Performance metrics
        if self.metrics_history:
            amounts = [m.amount for m in self.metrics_history if m.amount > 0]
            if amounts:
                report['performance_metrics'] = {
                    'total_volume_xrp': round(sum(amounts), 2),
                    'median_amount_xrp': round(statistics.median(amounts), 2),
                    'max_amount_xrp': round(max(amounts), 2),
                    'anomaly_rate_percent': round(
                        (self.stats['total_anomalies'] / self.stats['total_transactions']) * 100, 2
                    ) if self.stats['total_transactions'] > 0 else 0
                }
        
        return report
    
    def save_report(self, filename: str = None):
        """Save analytics report to JSON file"""
        if filename is None:
            filename = f"xrpl_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        report = self.generate_analytics_report()
        with open(filename, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"Report saved to {filename}")
        return filename


async def main():
    """Main function to run the monitor"""
    monitor = XRPLMonitor(
        websocket_url="wss://xrplcluster.com/",
        window_size=100,
        anomaly_threshold=2.5
    )
    
    try:
        # Start monitoring (will run indefinitely)
        await monitor.start_monitoring()
    except KeyboardInterrupt:
        logger.info("Monitoring stopped by user")
    except Exception as e:
        logger.error(f"Error in monitoring: {e}")
    finally:
        # Generate final report
        report = monitor.generate_analytics_report()
        print("\n" + "="*60)
        print("FINAL ANALYTICS REPORT")
        print("="*60)
        print(json.dumps(report, indent=2))
        
        # Save report
        monitor.save_report()


if __name__ == "__main__":
    asyncio.run(main())
