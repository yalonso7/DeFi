// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

interface IERC20 {
function transfer(address to, uint256 amount) external returns (bool);
function transferFrom(address from, address to, uint256 amount) external returns (bool);
function approve(address spender, uint256 amount) external returns (bool);
function balanceOf(address account) external view returns (uint256);
}

interface ILendingProtocol {
function deposit(uint256 amount) external returns (uint256);
function withdraw(uint256 amount) external returns (uint256);
function getAPY() external view returns (uint256);
function balanceOf(address account) external view returns (uint256);
}

/**

- @title YieldFarmingOptimizer
- @notice Automatically moves funds between lending protocols to maximize yield
- @dev Supports multiple protocols and rebalancing based on APY differences
  */
  contract YieldFarmingOptimizer {
  address public owner;
  IERC20 public baseToken;
  
  struct Protocol {
  address protocolAddress;
  string name;
  bool isActive;
  uint256 allocatedAmount;
  }
  
  mapping(uint256 => Protocol) public protocols;
  uint256 public protocolCount;
  
  uint256 public totalDeposits;
  uint256 public minRebalanceThreshold = 50; // 0.5% APY difference (in basis points)
  uint256 public rebalanceInterval = 1 days;
  uint256 public lastRebalance;
  
  mapping(address => uint256) public userDeposits;
  mapping(address => uint256) public userShares;
  uint256 public totalShares;
  
  uint256 public performanceFee = 1000; // 10% (in basis points)
  uint256 public constant MAX_FEE = 2000; // 20% max
  uint256 public accumulatedFees;
  
  event Deposited(address indexed user, uint256 amount, uint256 shares);
  event Withdrawn(address indexed user, uint256 amount, uint256 shares);
  event Rebalanced(uint256 indexed fromProtocol, uint256 indexed toProtocol, uint256 amount);
  event ProtocolAdded(uint256 indexed protocolId, address protocolAddress, string name);
  event ProtocolRemoved(uint256 indexed protocolId);
  event FeesCollected(uint256 amount);
  
  modifier onlyOwner() {
  require(msg.sender == owner, “Not owner”);
  _;
  }
  
  constructor(address _baseToken) {
  owner = msg.sender;
  baseToken = IERC20(_baseToken);
  lastRebalance = block.timestamp;
  }
  
  /**
  - @notice Add a new lending protocol to the strategy
    */
    function addProtocol(address _protocolAddress, string memory _name) external onlyOwner {
    protocols[protocolCount] = Protocol({
    protocolAddress: _protocolAddress,
    name: _name,
    isActive: true,
    allocatedAmount: 0
    });
    
    emit ProtocolAdded(protocolCount, _protocolAddress, _name);
    protocolCount++;
    }
  
  /**
  - @notice Deactivate a protocol (won’t be used for new deposits)
    */
    function deactivateProtocol(uint256 _protocolId) external onlyOwner {
    require(_protocolId < protocolCount, “Invalid protocol”);
    protocols[_protocolId].isActive = false;
    emit ProtocolRemoved(_protocolId);
    }
  
  /**
  - @notice User deposits funds into the optimizer
    */
    function deposit(uint256 _amount) external {
    require(_amount > 0, “Amount must be > 0”);
    require(baseToken.transferFrom(msg.sender, address(this), _amount), “Transfer failed”);
    
    uint256 shares;
    if (totalShares == 0) {
    shares = _amount;
    } else {
    shares = (_amount * totalShares) / totalDeposits;
    }
    
    userDeposits[msg.sender] += _amount;
    userShares[msg.sender] += shares;
    totalShares += shares;
    totalDeposits += _amount;
    
    // Deposit into best protocol
    uint256 bestProtocol = getBestProtocol();
    _depositToProtocol(bestProtocol, _amount);
    
    emit Deposited(msg.sender, _amount, shares);
    }
  
  /**
  - @notice User withdraws funds from the optimizer
    */
    function withdraw(uint256 _shares) external {
    require(_shares > 0 && _shares <= userShares[msg.sender], “Invalid shares”);
    
    uint256 totalValue = getTotalValue();
    uint256 withdrawAmount = (_shares * totalValue) / totalShares;
    
    // Withdraw from protocols proportionally
    _withdrawFromProtocols(withdrawAmount);
    
    userShares[msg.sender] -= _shares;
    totalShares -= _shares;
    totalDeposits -= withdrawAmount;
    
    require(baseToken.transfer(msg.sender, withdrawAmount), “Transfer failed”);
    
    emit Withdrawn(msg.sender, withdrawAmount, _shares);
    }
  
  /**
  - @notice Rebalance funds across protocols to maximize yield
    */
    function rebalance() external {
    require(block.timestamp >= lastRebalance + rebalanceInterval, “Too soon”);
    
    (uint256 bestProtocolId, uint256 bestAPY) = _getBestProtocolWithAPY();
    
    // Check all protocols and move funds if APY difference is significant
    for (uint256 i = 0; i < protocolCount; i++) {
    if (!protocols[i].isActive || i == bestProtocolId) continue;
    
    ```
     uint256 currentAPY = ILendingProtocol(protocols[i].protocolAddress).getAPY();
     
     // If APY difference exceeds threshold, move funds
     if (bestAPY > currentAPY + minRebalanceThreshold) {
         uint256 amountToMove = protocols[i].allocatedAmount;
         
         if (amountToMove > 0) {
             // Withdraw from current protocol
             ILendingProtocol(protocols[i].protocolAddress).withdraw(amountToMove);
             protocols[i].allocatedAmount = 0;
             
             // Deposit to best protocol
             _depositToProtocol(bestProtocolId, amountToMove);
             
             emit Rebalanced(i, bestProtocolId, amountToMove);
         }
     }
    ```
    
    }
    
    lastRebalance = block.timestamp;
    }
  
  /**
  - @notice Get the protocol with the highest APY
    */
    function getBestProtocol() public view returns (uint256) {
    (uint256 bestId, ) = _getBestProtocolWithAPY();
    return bestId;
    }
  
  /**
  - @notice Get total value locked in all protocols
    */
    function getTotalValue() public view returns (uint256) {
    uint256 total = baseToken.balanceOf(address(this));
    
    for (uint256 i = 0; i < protocolCount; i++) {
    if (protocols[i].isActive) {
    total += ILendingProtocol(protocols[i].protocolAddress).balanceOf(address(this));
    }
    }
    
    return total;
    }
  
  /**
  - @notice Get user’s current value
    */
    function getUserValue(address _user) external view returns (uint256) {
    if (totalShares == 0) return 0;
    return (userShares[_user] * getTotalValue()) / totalShares;
    }
  
  /**
  - @notice Get all protocol APYs
    */
    function getAllAPYs() external view returns (uint256[] memory) {
    uint256[] memory apys = new uint256[](protocolCount);
    
    for (uint256 i = 0; i < protocolCount; i++) {
    if (protocols[i].isActive) {
    apys[i] = ILendingProtocol(protocols[i].protocolAddress).getAPY();
    }
    }
    
    return apys;
    }
  
  /**
  - @notice Collect accumulated performance fees
    */
    function collectFees() external onlyOwner {
    uint256 fees = accumulatedFees;
    accumulatedFees = 0;
    
    require(baseToken.transfer(owner, fees), “Transfer failed”);
    emit FeesCollected(fees);
    }
  
  /**
  - @notice Update performance fee (max 20%)
    */
    function setPerformanceFee(uint256 _fee) external onlyOwner {
    require(_fee <= MAX_FEE, “Fee too high”);
    performanceFee = _fee;
    }
  
  /**
  - @notice Update rebalance threshold
    */
    function setRebalanceThreshold(uint256 _threshold) external onlyOwner {
    minRebalanceThreshold = _threshold;
    }
  
  /**
  - @notice Update rebalance interval
    */
    function setRebalanceInterval(uint256 _interval) external onlyOwner {
    rebalanceInterval = _interval;
    }
  
  // Internal functions
  
  function _depositToProtocol(uint256 _protocolId, uint256 _amount) internal {
  require(_protocolId < protocolCount, “Invalid protocol”);
  
  ```
   address protocolAddr = protocols[_protocolId].protocolAddress;
   baseToken.approve(protocolAddr, _amount);
   ILendingProtocol(protocolAddr).deposit(_amount);
   protocols[_protocolId].allocatedAmount += _amount;
  ```
  
  }
  
  function _withdrawFromProtocols(uint256 _amount) internal {
  uint256 remaining = _amount;
  
  ```
   // Withdraw proportionally from all active protocols
   for (uint256 i = 0; i < protocolCount && remaining > 0; i++) {
       if (!protocols[i].isActive || protocols[i].allocatedAmount == 0) continue;
       
       uint256 toWithdraw = (protocols[i].allocatedAmount * _amount) / getTotalValue();
       
       if (toWithdraw > remaining) {
           toWithdraw = remaining;
       }
       
       if (toWithdraw > 0) {
           ILendingProtocol(protocols[i].protocolAddress).withdraw(toWithdraw);
           protocols[i].allocatedAmount -= toWithdraw;
           remaining -= toWithdraw;
       }
   }
  ```
  
  }
  
  function _getBestProtocolWithAPY() internal view returns (uint256 bestId, uint256 bestAPY) {
  bestAPY = 0;
  bestId = 0;
  
  ```
   for (uint256 i = 0; i < protocolCount; i++) {
       if (!protocols[i].isActive) continue;
       
       uint256 apy = ILendingProtocol(protocols[i].protocolAddress).getAPY();
       if (apy > bestAPY) {
           bestAPY = apy;
           bestId = i;
       }
   }
  ```
  
  }
  
  /**
  - @notice Emergency withdraw all funds to owner
    */
    function emergencyWithdraw() external onlyOwner {
    for (uint256 i = 0; i < protocolCount; i++) {
    if (protocols[i].allocatedAmount > 0) {
    ILendingProtocol(protocols[i].protocolAddress).withdraw(
    protocols[i].allocatedAmount
    );
    protocols[i].allocatedAmount = 0;
    }
    }
    
    uint256 balance = baseToken.balanceOf(address(this));
    require(baseToken.transfer(owner, balance), “Transfer failed”);
    }
    }
