// ===================================================================
// COMPLETE WEB3 INTEGRATION FOR IDENTITY DAPP
// ===================================================================

// 1. CONTRACT ABI CONFIGURATION
// ===================================================================
const IDENTITY_ABI = [
  "function createIdentity(string memory _name, string memory _email) external",
  "function updateIdentity(string memory _name, string memory _email) external",
  "function addAttribute(string memory _key, string memory _value) external",
  "function getAttribute(address _owner, string memory _key) external view returns (string)",
  "function verifyIdentity(address _identity) external",
  "function requestAccess(address _owner, string memory _purpose) external",
  "function grantAccess(address _accessor) external",
  "function revokeAccess(address _accessor) external",
  "function hasAccess(address _owner, address _accessor) external view returns (bool)",
  "function getIdentity(address _owner) external view returns (address, string, string, bool, uint256, uint256)",
  "function getAccessRequests() external view returns (tuple(address requester, uint256 timestamp, bool approved, string purpose)[])",
  "function isVerified(address _owner) external view returns (bool)",
  "event IdentityCreated(address indexed owner, uint256 timestamp)",
  "event IdentityUpdated(address indexed owner, uint256 timestamp)",
  "event AccessGranted(address indexed owner, address indexed accessor)",
  "event AccessRevoked(address indexed owner, address indexed accessor)"
];

// Contract addresses for different networks
const CONTRACT_ADDRESSES = {
  mainnet: "0x0000000000000000000000000000000000000000",
  sepolia: "0x0000000000000000000000000000000000000000",
  polygon: "0x0000000000000000000000000000000000000000",
  localhost: "0x5FbDB2315678afecb367f032d93F642f64180aa3"
};

// 2. WEB3 PROVIDER SETUP WITH ETHERS.JS
// ===================================================================
import { ethers } from 'ethers';

class Web3Service {
  constructor() {
    this.provider = null;
    this.signer = null;
    this.contract = null;
    this.account = null;
    this.chainId = null;
  }

  // Initialize Web3 connection
  async connect() {
    try {
      if (typeof window.ethereum === 'undefined') {
        throw new Error('MetaMask is not installed');
      }

      // Request account access
      const accounts = await window.ethereum.request({
        method: 'eth_requestAccounts'
      });

      // Create provider and signer
      this.provider = new ethers.BrowserProvider(window.ethereum);
      this.signer = await this.provider.getSigner();
      this.account = accounts[0];

      // Get network
      const network = await this.provider.getNetwork();
      this.chainId = Number(network.chainId);

      // Initialize contract
      const contractAddress = this.getContractAddress();
      this.contract = new ethers.Contract(
        contractAddress,
        IDENTITY_ABI,
        this.signer
      );

      // Listen to account changes
      window.ethereum.on('accountsChanged', (accounts) => {
        this.account = accounts[0];
        window.location.reload();
      });

      // Listen to chain changes
      window.ethereum.on('chainChanged', () => {
        window.location.reload();
      });

      return {
        account: this.account,
        chainId: this.chainId,
        success: true
      };
    } catch (error) {
      console.error('Connection error:', error);
      throw error;
    }
  }

  // Get contract address based on chain
  getContractAddress() {
    const networkMap = {
      1: 'mainnet',
      11155111: 'sepolia',
      137: 'polygon',
      31337: 'localhost'
    };
    
    const network = networkMap[this.chainId] || 'localhost';
    return CONTRACT_ADDRESSES[network];
  }

  // Disconnect wallet
  disconnect() {
    this.provider = null;
    this.signer = null;
    this.contract = null;
    this.account = null;
  }

  // Get current account
  getAccount() {
    return this.account;
  }

  // Switch network
  async switchNetwork(chainId) {
    try {
      await window.ethereum.request({
        method: 'wallet_switchEthereumChain',
        params: [{ chainId: `0x${chainId.toString(16)}` }],
      });
    } catch (error) {
      console.error('Network switch error:', error);
      throw error;
    }
  }
}

// 3. IDENTITY CONTRACT INTERACTIONS
// ===================================================================
class IdentityService extends Web3Service {
  
  // Create a new identity
  async createIdentity(name, email) {
    try {
      const tx = await this.contract.createIdentity(name, email);
      const receipt = await tx.wait();
      
      // Parse events
      const event = receipt.logs.find(
        log => log.fragment && log.fragment.name === 'IdentityCreated'
      );
      
      return {
        success: true,
        transactionHash: receipt.hash,
        blockNumber: receipt.blockNumber,
        event: event
      };
    } catch (error) {
      console.error('Create identity error:', error);
      throw this.parseError(error);
    }
  }

  // Update existing identity
  async updateIdentity(name, email) {
    try {
      const tx = await this.contract.updateIdentity(name, email);
      const receipt = await tx.wait();
      
      return {
        success: true,
        transactionHash: receipt.hash,
        blockNumber: receipt.blockNumber
      };
    } catch (error) {
      console.error('Update identity error:', error);
      throw this.parseError(error);
    }
  }

  // Add custom attribute
  async addAttribute(key, value) {
    try {
      const tx = await this.contract.addAttribute(key, value);
      const receipt = await tx.wait();
      
      return {
        success: true,
        transactionHash: receipt.hash
      };
    } catch (error) {
      console.error('Add attribute error:', error);
      throw this.parseError(error);
    }
  }

  // Get attribute value
  async getAttribute(ownerAddress, key) {
    try {
      const value = await this.contract.getAttribute(ownerAddress, key);
      return value;
    } catch (error) {
      console.error('Get attribute error:', error);
      throw this.parseError(error);
    }
  }

  // Get identity information
  async getIdentity(address) {
    try {
      const [owner, name, email, isVerified, createdAt, updatedAt] = 
        await this.contract.getIdentity(address || this.account);
      
      return {
        owner,
        name,
        email,
        isVerified,
        createdAt: Number(createdAt),
        updatedAt: Number(updatedAt)
      };
    } catch (error) {
      console.error('Get identity error:', error);
      return null;
    }
  }

  // Check if identity exists
  async hasIdentity(address) {
    try {
      const identity = await this.getIdentity(address);
      return identity !== null;
    } catch (error) {
      return false;
    }
  }

  // Request access to another identity
  async requestAccess(ownerAddress, purpose) {
    try {
      const tx = await this.contract.requestAccess(ownerAddress, purpose);
      const receipt = await tx.wait();
      
      return {
        success: true,
        transactionHash: receipt.hash
      };
    } catch (error) {
      console.error('Request access error:', error);
      throw this.parseError(error);
    }
  }

  // Grant access to requester
  async grantAccess(accessorAddress) {
    try {
      const tx = await this.contract.grantAccess(accessorAddress);
      const receipt = await tx.wait();
      
      return {
        success: true,
        transactionHash: receipt.hash
      };
    } catch (error) {
      console.error('Grant access error:', error);
      throw this.parseError(error);
    }
  }

  // Revoke access
  async revokeAccess(accessorAddress) {
    try {
      const tx = await this.contract.revokeAccess(accessorAddress);
      const receipt = await tx.wait();
      
      return {
        success: true,
        transactionHash: receipt.hash
      };
    } catch (error) {
      console.error('Revoke access error:', error);
      throw this.parseError(error);
    }
  }

  // Check if address has access
  async hasAccess(ownerAddress, accessorAddress) {
    try {
      return await this.contract.hasAccess(ownerAddress, accessorAddress);
    } catch (error) {
      console.error('Has access error:', error);
      return false;
    }
  }

  // Get all access requests
  async getAccessRequests() {
    try {
      const requests = await this.contract.getAccessRequests();
      return requests.map(req => ({
        requester: req.requester,
        timestamp: Number(req.timestamp),
        approved: req.approved,
        purpose: req.purpose
      }));
    } catch (error) {
      console.error('Get access requests error:', error);
      return [];
    }
  }

  // Check if identity is verified
  async isVerified(address) {
    try {
      return await this.contract.isVerified(address || this.account);
    } catch (error) {
      console.error('Is verified error:', error);
      return false;
    }
  }

  // Listen to events
  listenToEvents(callback) {
    // Identity Created
    this.contract.on('IdentityCreated', (owner, timestamp, event) => {
      callback({
        type: 'IdentityCreated',
        owner,
        timestamp: Number(timestamp),
        event
      });
    });

    // Identity Updated
    this.contract.on('IdentityUpdated', (owner, timestamp, event) => {
      callback({
        type: 'IdentityUpdated',
        owner,
        timestamp: Number(timestamp),
        event
      });
    });

    // Access Granted
    this.contract.on('AccessGranted', (owner, accessor, event) => {
      callback({
        type: 'AccessGranted',
        owner,
        accessor,
        event
      });
    });

    // Access Revoked
    this.contract.on('AccessRevoked', (owner, accessor, event) => {
      callback({
        type: 'AccessRevoked',
        owner,
        accessor,
        event
      });
    });
  }

  // Remove event listeners
  removeAllListeners() {
    this.contract.removeAllListeners();
  }

  // Parse contract errors
  parseError(error) {
    if (error.reason) {
      return new Error(error.reason);
    }
    if (error.data && error.data.message) {
      return new Error(error.data.message);
    }
    if (error.message) {
      return new Error(error.message);
    }
    return new Error('Transaction failed');
  }

  // Estimate gas for transaction
  async estimateGas(functionName, ...args) {
    try {
      const gasEstimate = await this.contract[functionName].estimateGas(...args);
      return Number(gasEstimate);
    } catch (error) {
      console.error('Gas estimation error:', error);
      return null;
    }
  }

  // Get gas price
  async getGasPrice() {
    try {
      const feeData = await this.provider.getFeeData();
      return {
        gasPrice: ethers.formatUnits(feeData.gasPrice, 'gwei'),
        maxFeePerGas: ethers.formatUnits(feeData.maxFeePerGas, 'gwei'),
        maxPriorityFeePerGas: ethers.formatUnits(feeData.maxPriorityFeePerGas, 'gwei')
      };
    } catch (error) {
      console.error('Gas price error:', error);
      return null;
    }
  }
}

// 4. REACT HOOKS FOR IDENTITY MANAGEMENT
// ===================================================================

// Custom hook for identity service
export function useIdentityService() {
  const [service] = useState(() => new IdentityService());
  const [connected, setConnected] = useState(false);
  const [account, setAccount] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const connect = async () => {
    setLoading(true);
    setError(null);
    try {
      const result = await service.connect();
      setAccount(result.account);
      setConnected(true);
      return result;
    } catch (err) {
      setError(err.message);
      throw err;
    } finally {
      setLoading(false);
    }
  };

  const disconnect = () => {
    service.disconnect();
    setConnected(false);
    setAccount(null);
  };

  return {
    service,
    connected,
    account,
    loading,
    error,
    connect,
    disconnect
  };
}

// Custom hook for identity operations
export function useIdentity(service) {
  const [identity, setIdentity] = useState(null);
  const [loading, setLoading] = useState(false);

  const loadIdentity = async (address) => {
    setLoading(true);
    try {
      const data = await service.getIdentity(address);
      setIdentity(data);
      return data;
    } catch (error) {
      console.error(error);
      return null;
    } finally {
      setLoading(false);
    }
  };

  const createIdentity = async (name, email) => {
    setLoading(true);
    try {
      const result = await service.createIdentity(name, email);
      await loadIdentity();
      return result;
    } catch (error) {
      throw error;
    } finally {
      setLoading(false);
    }
  };

  const updateIdentity = async (name, email) => {
    setLoading(true);
    try {
      const result = await service.updateIdentity(name, email);
      await loadIdentity();
      return result;
    } catch (error) {
      throw error;
    } finally {
      setLoading(false);
    }
  };

  return {
    identity,
    loading,
    loadIdentity,
    createIdentity,
    updateIdentity
  };
}

// 5. BACKEND API INTEGRATION (NODE.JS/EXPRESS)
// ===================================================================

// Express API endpoints for off-chain operations
class IdentityAPI {
  constructor(app, identityService) {
    this.app = app;
    this.service = identityService;
    this.setupRoutes();
  }

  setupRoutes() {
    // Get identity by address
    this.app.get('/api/identity/:address', async (req, res) => {
      try {
        const identity = await this.service.getIdentity(req.params.address);
        res.json({ success: true, data: identity });
      } catch (error) {
        res.status(500).json({ success: false, error: error.message });
      }
    });

    // Verify identity status
    this.app.get('/api/identity/:address/verified', async (req, res) => {
      try {
        const isVerified = await this.service.isVerified(req.params.address);
        res.json({ success: true, verified: isVerified });
      } catch (error) {
        res.status(500).json({ success: false, error: error.message });
      }
    });

    // Get access requests
    this.app.get('/api/identity/:address/requests', async (req, res) => {
      try {
        const requests = await this.service.getAccessRequests();
        res.json({ success: true, data: requests });
      } catch (error) {
        res.status(500).json({ success: false, error: error.message });
      }
    });

    // Cache identity data (off-chain)
    this.app.post('/api/cache/identity', async (req, res) => {
      try {
        const { address, data } = req.body;
        // Store in database
        await this.cacheIdentity(address, data);
        res.json({ success: true });
      } catch (error) {
        res.status(500).json({ success: false, error: error.message });
      }
    });
  }

  async cacheIdentity(address, data) {
    // Implementation for caching identity data
    // Could use Redis, MongoDB, etc.
  }
}

// 6. UTILITY FUNCTIONS
// ===================================================================

// Format address for display
export function formatAddress(address) {
  if (!address) return '';
  return `${address.substring(0, 6)}...${address.substring(38)}`;
}

// Validate Ethereum address
export function isValidAddress(address) {
  return /^0x[a-fA-F0-9]{40}$/.test(address);
}

// Calculate transaction cost
export function calculateTransactionCost(gasUsed, gasPrice) {
  return ethers.formatEther(BigInt(gasUsed) * BigInt(gasPrice));
}

// Wait for transaction confirmation
export async function waitForConfirmation(provider, txHash, confirmations = 2) {
  const receipt = await provider.waitForTransaction(txHash, confirmations);
  return receipt;
}

// Export service instance
export const identityService = new IdentityService();
export default IdentityService;