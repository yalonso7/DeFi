// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

/**

- @title Cyfer Multi-Signature Wallet
- @notice A secure wallet requiring multiple signatures to approve transactions
- @dev Foundational DeFi primitive for managing shared funds with consensus
  */
  contract Cyfer {
  
  // Events
  event Deposit(address indexed sender, uint256 amount, uint256 balance);
  event SubmitTransaction(
  address indexed owner,
  uint256 indexed txIndex,
  address indexed to,
  uint256 value,
  bytes data
  );
  event ConfirmTransaction(address indexed owner, uint256 indexed txIndex);
  event RevokeConfirmation(address indexed owner, uint256 indexed txIndex);
  event ExecuteTransaction(address indexed owner, uint256 indexed txIndex);
  event OwnerAdded(address indexed owner);
  event OwnerRemoved(address indexed owner);
  event RequirementChanged(uint256 required);
  
  // State variables
  address[] public owners;
  mapping(address => bool) public isOwner;
  uint256 public numConfirmationsRequired;
  
  struct Transaction {
  address to;
  uint256 value;
  bytes data;
  bool executed;
  uint256 numConfirmations;
  uint256 timestamp;
  string description;
  }
  
  // Mapping from tx index => owner => bool
  mapping(uint256 => mapping(address => bool)) public isConfirmed;
  
  Transaction[] public transactions;
  
  // Modifiers
  modifier onlyOwner() {
  require(isOwner[msg.sender], “Not owner”);
  _;
  }
  
  modifier txExists(uint256 _txIndex) {
  require(_txIndex < transactions.length, “Tx does not exist”);
  _;
  }
  
  modifier notExecuted(uint256 _txIndex) {
  require(!transactions[_txIndex].executed, “Tx already executed”);
  _;
  }
  
  modifier notConfirmed(uint256 _txIndex) {
  require(!isConfirmed[_txIndex][msg.sender], “Tx already confirmed”);
  _;
  }
  
  /**
  - @notice Initialize the multi-sig wallet
  - @param _owners Array of owner addresses
  - @param _numConfirmationsRequired Number of confirmations needed
    */
    constructor(address[] memory _owners, uint256 _numConfirmationsRequired) {
    require(_owners.length > 0, “Owners required”);
    require(
    _numConfirmationsRequired > 0 &&
    _numConfirmationsRequired <= _owners.length,
    “Invalid number of confirmations”
    );
    
    for (uint256 i = 0; i < _owners.length; i++) {
    address owner = _owners[i];
    
    ```
     require(owner != address(0), "Invalid owner");
     require(!isOwner[owner], "Owner not unique");
     
     isOwner[owner] = true;
     owners.push(owner);
    ```
    
    }
    
    numConfirmationsRequired = _numConfirmationsRequired;
    }
  
  /**
  - @notice Receive ETH deposits
    */
    receive() external payable {
    emit Deposit(msg.sender, msg.value, address(this).balance);
    }
  
  /**
  - @notice Submit a new transaction for approval
  - @param _to Destination address
  - @param _value ETH value to send
  - @param _data Call data
  - @param _description Human-readable description
    */
    function submitTransaction(
    address _to,
    uint256 _value,
    bytes memory _data,
    string memory _description
    ) public onlyOwner {
    uint256 txIndex = transactions.length;
    
    transactions.push(
    Transaction({
    to: _to,
    value: _value,
    data: _data,
    executed: false,
    numConfirmations: 0,
    timestamp: block.timestamp,
    description: _description
    })
    );
    
    emit SubmitTransaction(msg.sender, txIndex, _to, _value, _data);
    }
  
  /**
  - @notice Confirm a pending transaction
  - @param _txIndex Transaction index
    */
    function confirmTransaction(uint256 _txIndex)
    public
    onlyOwner
    txExists(_txIndex)
    notExecuted(_txIndex)
    notConfirmed(_txIndex)
    {
    Transaction storage transaction = transactions[_txIndex];
    transaction.numConfirmations += 1;
    isConfirmed[_txIndex][msg.sender] = true;
    
    emit ConfirmTransaction(msg.sender, _txIndex);
    }
  
  /**
  - @notice Execute a confirmed transaction
  - @param _txIndex Transaction index
    */
    function executeTransaction(uint256 _txIndex)
    public
    onlyOwner
    txExists(_txIndex)
    notExecuted(_txIndex)
    {
    Transaction storage transaction = transactions[_txIndex];
    
    require(
    transaction.numConfirmations >= numConfirmationsRequired,
    “Cannot execute: insufficient confirmations”
    );
    
    transaction.executed = true;
    
    (bool success, ) = transaction.to.call{value: transaction.value}(
    transaction.data
    );
    require(success, “Tx failed”);
    
    emit ExecuteTransaction(msg.sender, _txIndex);
    }
  
  /**
  - @notice Revoke a confirmation
  - @param _txIndex Transaction index
    */
    function revokeConfirmation(uint256 _txIndex)
    public
    onlyOwner
    txExists(_txIndex)
    notExecuted(_txIndex)
    {
    require(isConfirmed[_txIndex][msg.sender], “Tx not confirmed”);
    
    Transaction storage transaction = transactions[_txIndex];
    transaction.numConfirmations -= 1;
    isConfirmed[_txIndex][msg.sender] = false;
    
    emit RevokeConfirmation(msg.sender, _txIndex);
    }
  
  /**
  - @notice Get all owners
    */
    function getOwners() public view returns (address[] memory) {
    return owners;
    }
  
  /**
  - @notice Get transaction count
    */
    function getTransactionCount() public view returns (uint256) {
    return transactions.length;
    }
  
  /**
  - @notice Get pending transaction count
    */
    function getPendingTransactionCount() public view returns (uint256) {
    uint256 count = 0;
    for (uint256 i = 0; i < transactions.length; i++) {
    if (!transactions[i].executed) {
    count++;
    }
    }
    return count;
    }
  
  /**
  - @notice Get transaction details
  - @param _txIndex Transaction index
    */
    function getTransaction(uint256 _txIndex)
    public
    view
    returns (
    address to,
    uint256 value,
    bytes memory data,
    bool executed,
    uint256 numConfirmations,
    uint256 timestamp,
    string memory description
    )
    {
    Transaction storage transaction = transactions[_txIndex];
    
    return (
    transaction.to,
    transaction.value,
    transaction.data,
    transaction.executed,
    transaction.numConfirmations,
    transaction.timestamp,
    transaction.description
    );
    }
  
  /**
  - @notice Check if owner has confirmed transaction
  - @param _txIndex Transaction index
  - @param _owner Owner address
    */
    function hasConfirmed(uint256 _txIndex, address _owner)
    public
    view
    returns (bool)
    {
    return isConfirmed[_txIndex][_owner];
    }
  
  /**
  - @notice Get all confirmations for a transaction
  - @param _txIndex Transaction index
    */
    function getConfirmations(uint256 _txIndex)
    public
    view
    returns (address[] memory)
    {
    address[] memory confirmations = new address[](owners.length);
    uint256 count = 0;
    
    for (uint256 i = 0; i < owners.length; i++) {
    if (isConfirmed[_txIndex][owners[i]]) {
    confirmations[count] = owners[i];
    count++;
    }
    }
    
    // Resize array to actual count
    address[] memory result = new address[](count);
    for (uint256 i = 0; i < count; i++) {
    result[i] = confirmations[i];
    }
    
    return result;
    }
  
  /**
  - @notice Get all pending transactions
    */
    function getPendingTransactions()
    public
    view
    returns (uint256[] memory)
    {
    uint256[] memory pending = new uint256[](transactions.length);
    uint256 count = 0;
    
    for (uint256 i = 0; i < transactions.length; i++) {
    if (!transactions[i].executed) {
    pending[count] = i;
    count++;
    }
    }
    
    // Resize array
    uint256[] memory result = new uint256[](count);
    for (uint256 i = 0; i < count; i++) {
    result[i] = pending[i];
    }
    
    return result;
    }
  
  /**
  - @notice Get transactions that need your confirmation
  - @param _owner Owner address
    */
    function getTransactionsNeedingConfirmation(address _owner)
    public
    view
    returns (uint256[] memory)
    {
    require(isOwner[_owner], “Not an owner”);
    
    uint256[] memory needConfirm = new uint256[](transactions.length);
    uint256 count = 0;
    
    for (uint256 i = 0; i < transactions.length; i++) {
    if (!transactions[i].executed && !isConfirmed[i][_owner]) {
    needConfirm[count] = i;
    count++;
    }
    }
    
    // Resize array
    uint256[] memory result = new uint256[](count);
    for (uint256 i = 0; i < count; i++) {
    result[i] = needConfirm[i];
    }
    
    return result;
    }
  
  /**
  - @notice Add a new owner (requires multi-sig approval)
  - @param _owner New owner address
    */
    function addOwner(address _owner) public onlyOwner {
    require(_owner != address(0), “Invalid owner”);
    require(!isOwner[_owner], “Owner already exists”);
    
    isOwner[_owner] = true;
    owners.push(_owner);
    
    emit OwnerAdded(_owner);
    }
  
  /**
  - @notice Remove an owner (requires multi-sig approval)
  - @param _owner Owner address to remove
    */
    function removeOwner(address _owner) public onlyOwner {
    require(isOwner[_owner], “Not an owner”);
    require(owners.length - 1 >= numConfirmationsRequired, “Cannot remove: would break requirement”);
    
    isOwner[_owner] = false;
    
    // Remove from array
    for (uint256 i = 0; i < owners.length; i++) {
    if (owners[i] == _owner) {
    owners[i] = owners[owners.length - 1];
    owners.pop();
    break;
    }
    }
    
    emit OwnerRemoved(_owner);
    }
  
  /**
  - @notice Change the number of required confirmations
  - @param _numConfirmationsRequired New requirement
    */
    function changeRequirement(uint256 _numConfirmationsRequired)
    public
    onlyOwner
    {
    require(
    _numConfirmationsRequired > 0 &&
    _numConfirmationsRequired <= owners.length,
    “Invalid requirement”
    );
    
    numConfirmationsRequired = _numConfirmationsRequired;
    
    emit RequirementChanged(_numConfirmationsRequired);
    }
  
  /**
  - @notice Get contract balance
    */
    function getBalance() public view returns (uint256) {
    return address(this).balance;
    }
  
  /**
  - @notice Submit and confirm a transaction in one call
  - @param _to Destination address
  - @param _value ETH value
  - @param _data Call data
  - @param _description Transaction description
    */
    function submitAndConfirm(
    address _to,
    uint256 _value,
    bytes memory _data,
    string memory _description
    ) external onlyOwner {
    uint256 txIndex = transactions.length;
    submitTransaction(_to, _value, _data, _description);
    confirmTransaction(txIndex);
    }
  
  /**
  - @notice Batch confirm multiple transactions
  - @param _txIndexes Array of transaction indexes
    */
    function batchConfirm(uint256[] memory _txIndexes) external onlyOwner {
    for (uint256 i = 0; i < _txIndexes.length; i++) {
    uint256 txIndex = _txIndexes[i];
    if (
    txIndex < transactions.length &&
    !transactions[txIndex].executed &&
    !isConfirmed[txIndex][msg.sender]
    ) {
    confirmTransaction(txIndex);
    }
    }
    }
  
  /**
  - @notice Check if transaction is ready to execute
  - @param _txIndex Transaction index
    */
    function isReadyToExecute(uint256 _txIndex)
    public
    view
    returns (bool)
    {
    if (_txIndex >= transactions.length) return false;
    
    Transaction storage transaction = transactions[_txIndex];
    
    return (
    !transaction.executed &&
    transaction.numConfirmations >= numConfirmationsRequired
    );
    }
    }
