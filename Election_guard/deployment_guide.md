# 🗳️ ElectionGuard DApp - Complete Deployment Guide

## 📋 Table of Contents
1. [System Architecture](#system-architecture)
2. [Prerequisites](#prerequisites)
3. [Backend Setup](#backend-setup)
4. [Blockchain Deployment](#blockchain-deployment)
5. [ML Model Training](#ml-model-training)
6. [Frontend Integration](#frontend-integration)
7. [API Documentation](#api-documentation)
8. [Security Considerations](#security-considerations)
9. [Monitoring & Maintenance](#monitoring--maintenance)

---

## 🏗️ System Architecture

```
┌─────────────────┐
│  React Frontend │
│   (Claude UI)   │
└────────┬────────┘
         │ HTTPS/REST
         ▼
┌─────────────────┐      ┌──────────────────┐
│  Flask Backend  │◄────►│  ML Engine       │
│  (Python API)   │      │  (TensorFlow/    │
└────────┬────────┘      │   PyTorch)       │
         │               └──────────────────┘
         │ Web3.py
         ▼
┌─────────────────┐      ┌──────────────────┐
│  Smart Contract │◄────►│  IPFS Storage    │
│  (Solidity)     │      │  (Optional)      │
└─────────────────┘      └──────────────────┘
         │
         ▼
┌─────────────────┐
│  Ethereum/L2    │
│  Blockchain     │
└─────────────────┘
```

---

## 🔧 Prerequisites

### Software Requirements
```bash
# System dependencies
Python 3.9+
Node.js 16+
npm or yarn
Ganache (for local blockchain)
MongoDB (optional, for caching)
Redis (optional, for queuing)

# Python packages
pip install flask flask-cors web3 tensorflow torch numpy pandas scipy scikit-learn

# Node packages (for frontend)
npm install react recharts lucide-react
```

### Hardware Requirements
- **Minimum**: 8GB RAM, 4 CPU cores, 100GB storage
- **Recommended**: 16GB RAM, 8 CPU cores, 500GB SSD

---

## 🚀 Backend Setup

### 1. Clone and Configure

```bash
# Create project directory
mkdir election-fraud-detection
cd election-fraud-detection

# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Environment Configuration

Create `.env` file:

```env
# Flask Configuration
FLASK_APP=election_fraud_detection_backend.py
FLASK_ENV=production
SECRET_KEY=your-secret-key-here

# Blockchain Configuration
WEB3_PROVIDER_URI=http://localhost:8545
CONTRACT_ADDRESS=0x...
CHAIN_ID=1337

# ML Model Paths
MODEL_PATH=./models/fraud_detection_model.pth
SCALER_PATH=./models/scaler.pkl
TRAINING_DATA_PATH=./data/training_data.csv

# Database (Optional)
MONGODB_URI=mongodb://localhost:27017/election_db
REDIS_URL=redis://localhost:6379

# API Configuration
API_RATE_LIMIT=100
MAX_UPLOAD_SIZE=50MB

# Security
ENABLE_CORS=true
ALLOWED_ORIGINS=http://localhost:3000,https://yourdomain.com
JWT_SECRET_KEY=your-jwt-secret
```

### 3. Initialize Database

```python
# initialize_db.py
from pymongo import MongoClient

client = MongoClient('mongodb://localhost:27017/')
db = client['election_db']

# Create collections
db.create_collection('elections')
db.create_collection('votes')
db.create_collection('anomalies')
db.create_collection('ml_predictions')

# Create indexes
db.votes.create_index([('voter_hash', 1), ('region_id', 1)], unique=True)
db.anomalies.create_index([('region_id', 1), ('timestamp', -1)])

print("Database initialized successfully!")
```

### 4. Start Backend Server

```bash
# Development mode
python election_fraud_detection_backend.py

# Production mode with Gunicorn
gunicorn -w 4 -b 0.0.0.0:5000 election_fraud_detection_backend:app

# With SSL (recommended for production)
gunicorn --certfile cert.pem --keyfile key.pem -w 4 -b 0.0.0.0:5000 election_fraud_detection_backend:app
```

---

## ⛓️ Blockchain Deployment

### 1. Local Development (Ganache)

```bash
# Install Ganache
npm install -g ganache

# Start Ganache
ganache --port 8545 --accounts 10 --networkId 1337
```

### 2. Deploy Smart Contract

```bash
# Install Truffle
npm install -g truffle

# Initialize Truffle project
truffle init

# Create deployment script (migrations/2_deploy_contracts.js)
```

```javascript
const ElectionVotingContract = artifacts.require("ElectionVotingContract");

module.exports = function(deployer) {
  deployer.deploy(ElectionVotingContract);
};
```

```bash
# Compile and deploy
truffle compile
truffle migrate --network development

# Save contract address and ABI
truffle networks
```

### 3. Production Deployment (Ethereum Mainnet/L2)

For production, consider:
- **Polygon**: Lower gas fees, faster transactions
- **Arbitrum**: Ethereum L2 with full EVM compatibility
- **Optimism**: Another popular L2 solution

```javascript
// truffle-config.js
module.exports = {
  networks: {
    polygon_mumbai: {
      provider: () => new HDWalletProvider(mnemonic, 
        `https://polygon-mumbai.g.alchemy.com/v2/${ALCHEMY_API_KEY}`),
      network_id: 80001,
      confirmations: 2,
      timeoutBlocks: 200,
      skipDryRun: true
    },
    polygon_mainnet: {
      provider: () => new HDWalletProvider(mnemonic,
        `https://polygon-mainnet.g.alchemy.com/v2/${ALCHEMY_API_KEY}`),
      network_id: 137,
      confirmations: 2,
      timeoutBlocks: 200,
      skipDryRun: true,
      gasPrice: 50000000000  // 50 gwei
    }
  }
};
```

### 4. Connect Backend to Blockchain

```python
# config.py
from web3 import Web3
import json

def connect_blockchain():
    w3 = Web3(Web3.HTTPProvider(os.getenv('WEB3_PROVIDER_URI')))
    
    # Load contract ABI
    with open('build/contracts/ElectionVotingContract.json', 'r')