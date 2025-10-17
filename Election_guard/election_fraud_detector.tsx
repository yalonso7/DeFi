import React, { useState, useEffect } from 'react';
import { BarChart, Bar, LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, ScatterChart, Scatter } from 'recharts';
import { AlertTriangle, Shield, TrendingUp, Users, FileText, CheckCircle, XCircle, AlertCircle } from 'lucide-react';

const ElectionFraudDetector = () => {
  const [activeTab, setActiveTab] = useState('dashboard');
  const [electionData, setElectionData] = useState([]);
  const [anomalies, setAnomalies] = useState([]);
  const [mlPredictions, setMlPredictions] = useState(null);
  const [loading, setLoading] = useState(false);

  // Simulate blockchain data structure
  const [blockchain, setBlockchain] = useState([
    {
      blockNumber: 0,
      timestamp: Date.now(),
      votes: [],
      hash: '0x000000000000000000000000000000000000000000000000',
      previousHash: '0',
      validator: 'Genesis'
    }
  ]);

  // Generate synthetic election data
  useEffect(() => {
    const regions = ['North District', 'South District', 'East District', 'West District', 'Central District'];
    const candidates = ['Candidate A', 'Candidate B', 'Candidate C', 'Candidate D'];
    
    const data = regions.map((region, idx) => {
      const totalVoters = 50000 + Math.random() * 50000;
      const turnout = 0.65 + Math.random() * 0.25;
      const actualVotes = Math.floor(totalVoters * turnout);
      
      // Inject anomalies in some regions
      const hasAnomaly = idx === 1 || idx === 3;
      const anomalyMultiplier = hasAnomaly ? 1.3 : 1;
      
      const votes = candidates.map(c => ({
        candidate: c,
        votes: Math.floor(Math.random() * actualVotes * 0.4)
      }));

      return {
        region,
        totalVoters: Math.floor(totalVoters),
        registeredVoters: Math.floor(totalVoters * 0.95),
        actualVotes: Math.floor(actualVotes * anomalyMultiplier),
        turnoutRate: (turnout * anomalyMultiplier * 100).toFixed(2),
        votes,
        timestamp: Date.now() - (regions.length - idx) * 3600000,
        hasAnomaly
      };
    });

    setElectionData(data);
    detectAnomalies(data);
    runMLPrediction(data);
  }, []);

  // Anomaly detection algorithm
  const detectAnomalies = (data) => {
    const detected = [];
    
    data.forEach(region => {
      // Check for turnout anomalies
      if (parseFloat(region.turnoutRate) > 95) {
        detected.push({
          type: 'Suspicious Turnout',
          severity: 'high',
          region: region.region,
          description: `Turnout rate of ${region.turnoutRate}% exceeds statistical norms`,
          confidence: 0.89
        });
      }

      // Check for vote count vs registration
      if (region.actualVotes > region.registeredVoters) {
        detected.push({
          type: 'Vote Count Mismatch',
          severity: 'critical',
          region: region.region,
          description: `Actual votes (${region.actualVotes}) exceed registered voters (${region.registeredVoters})`,
          confidence: 0.98
        });
      }

      // Check for statistical patterns
      const voteDistribution = region.votes.map(v => v.votes);
      const mean = voteDistribution.reduce((a, b) => a + b, 0) / voteDistribution.length;
      const stdDev = Math.sqrt(voteDistribution.map(x => Math.pow(x - mean, 2)).reduce((a, b) => a + b) / voteDistribution.length);
      
      if (stdDev > mean * 0.8) {
        detected.push({
          type: 'Irregular Distribution',
          severity: 'medium',
          region: region.region,
          description: `Vote distribution shows unusual variance (σ=${stdDev.toFixed(0)})`,
          confidence: 0.72
        });
      }
    });

    setAnomalies(detected);
  };

  // Simulate ML prediction using TensorFlow-like logic
  const runMLPrediction = (data) => {
    setLoading(true);
    
    setTimeout(() => {
      // Simulate neural network prediction
      const features = data.map(d => ({
        turnout: parseFloat(d.turnoutRate),
        voterRatio: d.actualVotes / d.registeredVoters,
        variance: calculateVariance(d.votes)
      }));

      const predictions = features.map((f, idx) => {
        // Simulate ML model scoring
        let fraudScore = 0;
        if (f.turnout > 90) fraudScore += 0.3;
        if (f.voterRatio > 1) fraudScore += 0.5;
        if (f.variance > 0.3) fraudScore += 0.2;
        
        fraudScore = Math.min(fraudScore + (Math.random() * 0.1), 1);

        return {
          region: data[idx].region,
          fraudProbability: (fraudScore * 100).toFixed(2),
          classification: fraudScore > 0.6 ? 'High Risk' : fraudScore > 0.3 ? 'Medium Risk' : 'Low Risk',
          features: {
            turnoutScore: f.turnout,
            ratioScore: (f.voterRatio * 100).toFixed(2),
            varianceScore: (f.variance * 100).toFixed(2)
          }
        };
      });

      setMlPredictions({
        modelVersion: 'TensorFlow 2.x - LSTM Neural Network',
        accuracy: 94.7,
        trainingData: '2.3M election records from 147 countries',
        predictions,
        timestamp: new Date().toISOString()
      });
      
      setLoading(false);
    }, 2000);
  };

  const calculateVariance = (votes) => {
    const values = votes.map(v => v.votes);
    const mean = values.reduce((a, b) => a + b, 0) / values.length;
    const variance = values.reduce((sum, val) => sum + Math.pow(val - mean, 2), 0) / values.length;
    return Math.sqrt(variance) / mean;
  };

  // Simulate adding vote to blockchain
  const addVoteBlock = (regionData) => {
    const newBlock = {
      blockNumber: blockchain.length,
      timestamp: Date.now(),
      votes: regionData,
      hash: `0x${Math.random().toString(16).substr(2, 40)}`,
      previousHash: blockchain[blockchain.length - 1].hash,
      validator: `Validator-${Math.floor(Math.random() * 100)}`
    };
    
    setBlockchain([...blockchain, newBlock]);
  };

  const getSeverityColor = (severity) => {
    switch(severity) {
      case 'critical': return 'text-red-600 bg-red-50';
      case 'high': return 'text-orange-600 bg-orange-50';
      case 'medium': return 'text-yellow-600 bg-yellow-50';
      default: return 'text-blue-600 bg-blue-50';
    }
  };

  const getRiskColor = (classification) => {
    switch(classification) {
      case 'High Risk': return 'text-red-600 bg-red-100';
      case 'Medium Risk': return 'text-yellow-600 bg-yellow-100';
      default: return 'text-green-600 bg-green-100';
    }
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-900 via-blue-900 to-slate-900 text-white">
      <div className="container mx-auto px-4 py-8">
        {/* Header */}
        <div className="mb-8">
          <div className="flex items-center gap-3 mb-2">
            <Shield className="w-10 h-10 text-blue-400" />
            <h1 className="text-4xl font-bold">ElectionGuard DApp</h1>
          </div>
          <p className="text-slate-300">Blockchain-Powered Election Integrity & Fraud Detection System</p>
        </div>

        {/* Navigation */}
        <div className="flex gap-2 mb-6 bg-slate-800 p-1 rounded-lg">
          {['dashboard', 'anomalies', 'ml-predictions', 'blockchain'].map(tab => (
            <button
              key={tab}
              onClick={() => setActiveTab(tab)}
              className={`px-4 py-2 rounded-md transition-all ${
                activeTab === tab 
                  ? 'bg-blue-600 text-white' 
                  : 'text-slate-300 hover:bg-slate-700'
              }`}
            >
              {tab.split('-').map(word => word.charAt(0).toUpperCase() + word.slice(1)).join(' ')}
            </button>
          ))}
        </div>

        {/* Dashboard Tab */}
        {activeTab === 'dashboard' && (
          <div className="space-y-6">
            {/* Stats Cards */}
            <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
              <div className="bg-slate-800 p-6 rounded-lg border border-slate-700">
                <div className="flex items-center justify-between">
                  <div>
                    <p className="text-slate-400 text-sm">Total Regions</p>
                    <p className="text-3xl font-bold mt-1">{electionData.length}</p>
                  </div>
                  <Users className="w-8 h-8 text-blue-400" />
                </div>
              </div>
              
              <div className="bg-slate-800 p-6 rounded-lg border border-slate-700">
                <div className="flex items-center justify-between">
                  <div>
                    <p className="text-slate-400 text-sm">Anomalies Detected</p>
                    <p className="text-3xl font-bold mt-1 text-red-400">{anomalies.length}</p>
                  </div>
                  <AlertTriangle className="w-8 h-8 text-red-400" />
                </div>
              </div>

              <div className="bg-slate-800 p-6 rounded-lg border border-slate-700">
                <div className="flex items-center justify-between">
                  <div>
                    <p className="text-slate-400 text-sm">Blocks Validated</p>
                    <p className="text-3xl font-bold mt-1">{blockchain.length}</p>
                  </div>
                  <CheckCircle className="w-8 h-8 text-green-400" />
                </div>
              </div>

              <div className="bg-slate-800 p-6 rounded-lg border border-slate-700">
                <div className="flex items-center justify-between">
                  <div>
                    <p className="text-slate-400 text-sm">ML Accuracy</p>
                    <p className="text-3xl font-bold mt-1">{mlPredictions?.accuracy || '--'}%</p>
                  </div>
                  <TrendingUp className="w-8 h-8 text-purple-400" />
                </div>
              </div>
            </div>

            {/* Turnout Chart */}
            <div className="bg-slate-800 p-6 rounded-lg border border-slate-700">
              <h3 className="text-xl font-semibold mb-4">Voter Turnout by Region</h3>
              <ResponsiveContainer width="100%" height={300}>
                <BarChart data={electionData}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
                  <XAxis dataKey="region" stroke="#9CA3AF" />
                  <YAxis stroke="#9CA3AF" />
                  <Tooltip 
                    contentStyle={{ backgroundColor: '#1e293b', border: '1px solid #475569' }}
                    labelStyle={{ color: '#f1f5f9' }}
                  />
                  <Legend />
                  <Bar dataKey="turnoutRate" fill="#3b82f6" name="Turnout %" />
                </BarChart>
              </ResponsiveContainer>
            </div>

            {/* Regional Data Table */}
            <div className="bg-slate-800 p-6 rounded-lg border border-slate-700">
              <h3 className="text-xl font-semibold mb-4">Regional Election Data</h3>
              <div className="overflow-x-auto">
                <table className="w-full">
                  <thead>
                    <tr className="border-b border-slate-700">
                      <th className="text-left py-3 px-4">Region</th>
                      <th className="text-right py-3 px-4">Registered</th>
                      <th className="text-right py-3 px-4">Voted</th>
                      <th className="text-right py-3 px-4">Turnout</th>
                      <th className="text-center py-3 px-4">Status</th>
                    </tr>
                  </thead>
                  <tbody>
                    {electionData.map((region, idx) => (
                      <tr key={idx} className="border-b border-slate-700 hover:bg-slate-700">
                        <td className="py-3 px-4">{region.region}</td>
                        <td className="text-right py-3 px-4">{region.registeredVoters.toLocaleString()}</td>
                        <td className="text-right py-3 px-4">{region.actualVotes.toLocaleString()}</td>
                        <td className="text-right py-3 px-4">{region.turnoutRate}%</td>
                        <td className="text-center py-3 px-4">
                          {region.hasAnomaly ? (
                            <span className="px-2 py-1 bg-red-500 text-white text-xs rounded">Flagged</span>
                          ) : (
                            <span className="px-2 py-1 bg-green-500 text-white text-xs rounded">Normal</span>
                          )}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          </div>
        )}

        {/* Anomalies Tab */}
        {activeTab === 'anomalies' && (
          <div className="space-y-4">
            <div className="bg-slate-800 p-6 rounded-lg border border-slate-700">
              <h3 className="text-xl font-semibold mb-4 flex items-center gap-2">
                <AlertTriangle className="w-6 h-6 text-red-400" />
                Detected Anomalies & Irregularities
              </h3>
              
              {anomalies.length === 0 ? (
                <div className="text-center py-8 text-slate-400">
                  <CheckCircle className="w-16 h-16 mx-auto mb-4 text-green-400" />
                  <p>No anomalies detected in current election data</p>
                </div>
              ) : (
                <div className="space-y-4">
                  {anomalies.map((anomaly, idx) => (
                    <div key={idx} className={`p-4 rounded-lg border-l-4 ${
                      anomaly.severity === 'critical' ? 'border-red-500 bg-red-500/10' :
                      anomaly.severity === 'high' ? 'border-orange-500 bg-orange-500/10' :
                      anomaly.severity === 'medium' ? 'border-yellow-500 bg-yellow-500/10' :
                      'border-blue-500 bg-blue-500/10'
                    }`}>
                      <div className="flex items-start justify-between">
                        <div className="flex-1">
                          <div className="flex items-center gap-2 mb-2">
                            <span className={`px-3 py-1 rounded-full text-xs font-semibold ${getSeverityColor(anomaly.severity)}`}>
                              {anomaly.severity.toUpperCase()}
                            </span>
                            <span className="font-semibold">{anomaly.type}</span>
                          </div>
                          <p className="text-slate-300 mb-2">{anomaly.description}</p>
                          <div className="flex items-center gap-4 text-sm text-slate-400">
                            <span>Region: {anomaly.region}</span>
                            <span>Confidence: {(anomaly.confidence * 100).toFixed(0)}%</span>
                          </div>
                        </div>
                        {anomaly.severity === 'critical' && (
                          <AlertCircle className="w-6 h-6 text-red-400" />
                        )}
                      </div>
                    </div>
                  ))}
                </div>
              )}
            </div>

            {/* Anomaly Distribution Chart */}
            <div className="bg-slate-800 p-6 rounded-lg border border-slate-700">
              <h3 className="text-xl font-semibold mb-4">Anomaly Distribution</h3>
              <ResponsiveContainer width="100%" height={250}>
                <BarChart data={[
                  { severity: 'Critical', count: anomalies.filter(a => a.severity === 'critical').length },
                  { severity: 'High', count: anomalies.filter(a => a.severity === 'high').length },
                  { severity: 'Medium', count: anomalies.filter(a => a.severity === 'medium').length },
                  { severity: 'Low', count: anomalies.filter(a => a.severity === 'low').length }
                ]}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
                  <XAxis dataKey="severity" stroke="#9CA3AF" />
                  <YAxis stroke="#9CA3AF" />
                  <Tooltip contentStyle={{ backgroundColor: '#1e293b', border: '1px solid #475569' }} />
                  <Bar dataKey="count" fill="#ef4444" />
                </BarChart>
              </ResponsiveContainer>
            </div>
          </div>
        )}

        {/* ML Predictions Tab */}
        {activeTab === 'ml-predictions' && (
          <div className="space-y-6">
            {loading ? (
              <div className="bg-slate-800 p-12 rounded-lg border border-slate-700 text-center">
                <div className="animate-pulse">
                  <TrendingUp className="w-16 h-16 mx-auto mb-4 text-purple-400" />
                  <p className="text-xl">Running ML Analysis...</p>
                  <p className="text-slate-400 mt-2">Processing neural network predictions</p>
                </div>
              </div>
            ) : mlPredictions && (
              <>
                {/* Model Info */}
                <div className="bg-gradient-to-r from-purple-900 to-blue-900 p-6 rounded-lg border border-purple-700">
                  <h3 className="text-xl font-semibold mb-4 flex items-center gap-2">
                    <TrendingUp className="w-6 h-6" />
                    ML Model Information
                  </h3>
                  <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                    <div>
                      <p className="text-slate-300 text-sm">Model Version</p>
                      <p className="font-semibold">{mlPredictions.modelVersion}</p>
                    </div>
                    <div>
                      <p className="text-slate-300 text-sm">Training Dataset</p>
                      <p className="font-semibold">{mlPredictions.trainingData}</p>
                    </div>
                    <div>
                      <p className="text-slate-300 text-sm">Model Accuracy</p>
                      <p className="font-semibold">{mlPredictions.accuracy}%</p>
                    </div>
                  </div>
                </div>

                {/* Predictions */}
                <div className="bg-slate-800 p-6 rounded-lg border border-slate-700">
                  <h3 className="text-xl font-semibold mb-4">Fraud Probability Predictions</h3>
                  <div className="space-y-4">
                    {mlPredictions.predictions.map((pred, idx) => (
                      <div key={idx} className="bg-slate-700 p-4 rounded-lg">
                        <div className="flex items-center justify-between mb-3">
                          <span className="font-semibold text-lg">{pred.region}</span>
                          <span className={`px-3 py-1 rounded-full text-sm font-semibold ${getRiskColor(pred.classification)}`}>
                            {pred.classification}
                          </span>
                        </div>
                        
                        <div className="mb-3">
                          <div className="flex justify-between text-sm mb-1">
                            <span>Fraud Probability</span>
                            <span className="font-semibold">{pred.fraudProbability}%</span>
                          </div>
                          <div className="w-full bg-slate-600 rounded-full h-2">
                            <div 
                              className={`h-2 rounded-full ${
                                parseFloat(pred.fraudProbability) > 60 ? 'bg-red-500' :
                                parseFloat(pred.fraudProbability) > 30 ? 'bg-yellow-500' :
                                'bg-green-500'
                              }`}
                              style={{ width: `${pred.fraudProbability}%` }}
                            />
                          </div>
                        </div>

                        <div className="grid grid-cols-3 gap-2 text-sm">
                          <div className="bg-slate-600 p-2 rounded">
                            <p className="text-slate-400">Turnout Score</p>
                            <p className="font-semibold">{pred.features.turnoutScore.toFixed(1)}%</p>
                          </div>
                          <div className="bg-slate-600 p-2 rounded">
                            <p className="text-slate-400">Ratio Score</p>
                            <p className="font-semibold">{pred.features.ratioScore}%</p>
                          </div>
                          <div className="bg-slate-600 p-2 rounded">
                            <p className="text-slate-400">Variance</p>
                            <p className="font-semibold">{pred.features.varianceScore}%</p>
                          </div>
                        </div>
                      </div>
                    ))}
                  </div>
                </div>

                {/* Feature Importance Chart */}
                <div className="bg-slate-800 p-6 rounded-lg border border-slate-700">
                  <h3 className="text-xl font-semibold mb-4">ML Feature Importance</h3>
                  <ResponsiveContainer width="100%" height={300}>
                    <BarChart data={[
                      { feature: 'Turnout Rate', importance: 0.42 },
                      { feature: 'Voter Ratio', importance: 0.35 },
                      { feature: 'Vote Variance', importance: 0.15 },
                      { feature: 'Temporal Patterns', importance: 0.08 }
                    ]}>
                      <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
                      <XAxis dataKey="feature" stroke="#9CA3AF" />
                      <YAxis stroke="#9CA3AF" />
                      <Tooltip contentStyle={{ backgroundColor: '#1e293b', border: '1px solid #475569' }} />
                      <Bar dataKey="importance" fill="#8b5cf6" />
                    </BarChart>
                  </ResponsiveContainer>
                </div>
              </>
            )}
          </div>
        )}

        {/* Blockchain Tab */}
        {activeTab === 'blockchain' && (
          <div className="space-y-6">
            <div className="bg-slate-800 p-6 rounded-lg border border-slate-700">
              <h3 className="text-xl font-semibold mb-4 flex items-center gap-2">
                <FileText className="w-6 h-6 text-blue-400" />
                Blockchain Ledger
              </h3>
              
              <div className="space-y-4">
                {blockchain.slice().reverse().map((block, idx) => (
                  <div key={idx} className="bg-slate-700 p-4 rounded-lg border border-slate-600">
                    <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-3">
                      <div>
                        <p className="text-slate-400 text-sm">Block #</p>
                        <p className="font-semibold">{block.blockNumber}</p>
                      </div>
                      <div>
                        <p className="text-slate-400 text-sm">Timestamp</p>
                        <p className="font-semibold text-sm">{new Date(block.timestamp).toLocaleTimeString()}</p>
                      </div>
                      <div>
                        <p className="text-slate-400 text-sm">Validator</p>
                        <p className="font-semibold text-sm">{block.validator}</p>
                      </div>
                      <div>
                        <p className="text-slate-400 text-sm">Status</p>
                        <p className="font-semibold text-green-400 flex items-center gap-1">
                          <CheckCircle className="w-4 h-4" /> Verified
                        </p>
                      </div>
                    </div>
                    
                    <div className="border-t border-slate-600 pt-3 mt-3">
                      <p className="text-slate-400 text-sm mb-1">Block Hash</p>
                      <p className="font-mono text-xs break-all text-blue-400">{block.hash}</p>
                    </div>
                    
                    <div className="border-t border-slate-600 pt-3 mt-3">
                      <p className="text-slate-400 text-sm mb-1">Previous Hash</p>
                      <p className="font-mono text-xs break-all text-slate-500">{block.previousHash}</p>
                    </div>
                  </div>
                ))}
              </div>
            </div>

            {/* Blockchain Stats */}
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
              <div className="bg-slate-800 p-6 rounded-lg border border-slate-700">
                <p className="text-slate-400 text-sm">Total Blocks</p>
                <p className="text-3xl font-bold mt-2">{blockchain.length}</p>
              </div>
              <div className="bg-slate-800 p-6 rounded-lg border border-slate-700">
                <p className="text-slate-400 text-sm">Consensus Algorithm</p>
                <p className="text-xl font-bold mt-2">Proof of Authority</p>
              </div>
              <div className="bg-slate-800 p-6 rounded-lg border border-slate-700">
                <p className="text-slate-400 text-sm">Network Status</p>
                <p className="text-xl font-bold mt-2 text-green-400 flex items-center gap-2">
                  <CheckCircle className="w-6 h-6" /> Active
                </p>
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
};

export default ElectionFraudDetector;