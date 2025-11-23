// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

/**

- @title UnifiedDAOGovernance
- @notice Complete DAO with governance token and voting system in one contract
- @dev Combines ERC20 voting token with proposal creation and execution
  */
  contract UnifiedDAOGovernance {
  
  // ============ TOKEN STATE ============
  
  string public name = “DAO Governance Token”;
  string public symbol = “DGOV”;
  uint8 public decimals = 18;
  uint256 public totalSupply;
  
  mapping(address => uint256) public balanceOf;
  mapping(address => mapping(address => uint256)) public allowance;
  
  // Vote delegation
  struct Checkpoint {
  uint256 fromBlock;
  uint256 votes;
  }
  
  mapping(address => Checkpoint[]) public checkpoints;
  mapping(address => address) public delegates;
  
  // ============ GOVERNANCE STATE ============
  
  enum ProposalState {
  Pending,
  Active,
  Canceled,
  Defeated,
  Succeeded,
  Queued,
  Expired,
  Executed
  }
  
  enum VoteType {
  Against,
  For,
  Abstain
  }
  
  struct Proposal {
  uint256 id;
  address proposer;
  string title;
  string description;
  address[] targets;
  uint256[] values;
  bytes[] calldatas;
  uint256 startBlock;
  uint256 endBlock;
  uint256 forVotes;
  uint256 againstVotes;
  uint256 abstainVotes;
  bool canceled;
  bool executed;
  uint256 eta;
  }
  
  struct Receipt {
  bool hasVoted;
  VoteType support;
  uint256 votes;
  }
  
  struct ProposalInfo {
  uint256 id;
  address proposer;
  string title;
  string description;
  uint256 startBlock;
  uint256 endBlock;
  uint256 forVotes;
  uint256 againstVotes;
  uint256 abstainVotes;
  bool canceled;
  bool executed;
  ProposalState state;
  }
  
  mapping(uint256 => Proposal) public proposals;
  mapping(uint256 => mapping(address => Receipt)) public receipts;
  uint256 public proposalCount;
  
  // Governance parameters
  uint256 public votingDelay = 1;
  uint256 public votingPeriod = 50400; // ~1 week
  uint256 public proposalThreshold = 1000 * 10**18;
  uint256 public quorumVotes = 4000 * 10**18;
  uint256 public timelockDelay = 2 days;
  
  address public admin;
  
  // ============ EVENTS ============
  
  // Token events
  event Transfer(address indexed from, address indexed to, uint256 value);
  event Approval(address indexed owner, address indexed spender, uint256 value);
  event DelegateChanged(address indexed delegator, address indexed fromDelegate, address indexed toDelegate);
  event DelegateVotesChanged(address indexed delegate, uint256 previousBalance, uint256 newBalance);
  
  // Governance events
  event ProposalCreated(uint256 id, address proposer, string title, uint256 startBlock, uint256 endBlock);
  event VoteCast(address indexed voter, uint256 proposalId, VoteType support, uint256 votes, string reason);
  event ProposalCanceled(uint256 id);
  event ProposalQueued(uint256 id, uint256 eta);
  event ProposalExecuted(uint256 id);
  
  // ============ MODIFIERS ============
  
  modifier onlyAdmin() {
  require(msg.sender == admin, “Not admin”);
  _;
  }
  
  // ============ CONSTRUCTOR ============
  
  constructor(uint256 _initialSupply) {
  admin = msg.sender;
  totalSupply = _initialSupply * 10**decimals;
  balanceOf[msg.sender] = totalSupply;
  emit Transfer(address(0), msg.sender, totalSupply);
  }
  
  // ============================================
  // TOKEN FUNCTIONS
  // ============================================
  
  /**
  - @notice Transfer tokens
    */
    function transfer(address _to, uint256 _value) external returns (bool) {
    require(balanceOf[msg.sender] >= _value, “Insufficient balance”);
    
    _moveVotingPower(delegates[msg.sender], delegates[_to], _value);
    
    balanceOf[msg.sender] -= _value;
    balanceOf[_to] += _value;
    
    emit Transfer(msg.sender, _to, _value);
    return true;
    }
  
  /**
  - @notice Approve spending
    */
    function approve(address _spender, uint256 _value) external returns (bool) {
    allowance[msg.sender][_spender] = _value;
    emit Approval(msg.sender, _spender, _value);
    return true;
    }
  
  /**
  - @notice Transfer from approved address
    */
    function transferFrom(address _from, address _to, uint256 _value) external returns (bool) {
    require(balanceOf[_from] >= _value, “Insufficient balance”);
    require(allowance[_from][msg.sender] >= _value, “Insufficient allowance”);
    
    _moveVotingPower(delegates[_from], delegates[_to], _value);
    
    balanceOf[_from] -= _value;
    balanceOf[_to] += _value;
    allowance[_from][msg.sender] -= _value;
    
    emit Transfer(_from, _to, _value);
    return true;
    }
  
  /**
  - @notice Delegate votes to another address
    */
    function delegate(address _delegatee) external {
    address currentDelegate = delegates[msg.sender];
    uint256 delegatorBalance = balanceOf[msg.sender];
    
    delegates[msg.sender] = _delegatee;
    
    emit DelegateChanged(msg.sender, currentDelegate, _delegatee);
    _moveVotingPower(currentDelegate, _delegatee, delegatorBalance);
    }
  
  /**
  - @notice Get current voting power
    */
    function getCurrentVotes(address _account) external view returns (uint256) {
    uint256 nCheckpoints = checkpoints[_account].length;
    return nCheckpoints > 0 ? checkpoints[_account][nCheckpoints - 1].votes : 0;
    }
  
  /**
  - @notice Get voting power at specific block
    */
    function getPriorVotes(address _account, uint256 _blockNumber) public view returns (uint256) {
    require(_blockNumber < block.number, “Not yet determined”);
    
    uint256 nCheckpoints = checkpoints[_account].length;
    if (nCheckpoints == 0) return 0;
    
    if (checkpoints[_account][nCheckpoints - 1].fromBlock <= _blockNumber) {
    return checkpoints[_account][nCheckpoints - 1].votes;
    }
    
    if (checkpoints[_account][0].fromBlock > _blockNumber) {
    return 0;
    }
    
    uint256 lower = 0;
    uint256 upper = nCheckpoints - 1;
    while (upper > lower) {
    uint256 center = upper - (upper - lower) / 2;
    Checkpoint memory cp = checkpoints[_account][center];
    if (cp.fromBlock == _blockNumber) {
    return cp.votes;
    } else if (cp.fromBlock < _blockNumber) {
    lower = center;
    } else {
    upper = center - 1;
    }
    }
    return checkpoints[_account][lower].votes;
    }
  
  function _moveVotingPower(address _from, address _to, uint256 _amount) internal {
  if (_from != _to && _amount > 0) {
  if (_from != address(0)) {
  uint256 nCheckpoints = checkpoints[_from].length;
  uint256 oldVotes = nCheckpoints > 0 ? checkpoints[_from][nCheckpoints - 1].votes : 0;
  uint256 newVotes = oldVotes - _amount;
  _writeCheckpoint(_from, oldVotes, newVotes);
  }
  
  ```
       if (_to != address(0)) {
           uint256 nCheckpoints = checkpoints[_to].length;
           uint256 oldVotes = nCheckpoints > 0 ? checkpoints[_to][nCheckpoints - 1].votes : 0;
           uint256 newVotes = oldVotes + _amount;
           _writeCheckpoint(_to, oldVotes, newVotes);
       }
   }
  ```
  
  }
  
  function _writeCheckpoint(address _delegatee, uint256 _oldVotes, uint256 _newVotes) internal {
  uint256 nCheckpoints = checkpoints[_delegatee].length;
  
  ```
   if (nCheckpoints > 0 && checkpoints[_delegatee][nCheckpoints - 1].fromBlock == block.number) {
       checkpoints[_delegatee][nCheckpoints - 1].votes = _newVotes;
   } else {
       checkpoints[_delegatee].push(Checkpoint(block.number, _newVotes));
   }
   
   emit DelegateVotesChanged(_delegatee, _oldVotes, _newVotes);
  ```
  
  }
  
  // ============================================
  // GOVERNANCE FUNCTIONS
  // ============================================
  
  /**
  - @notice Create a new proposal
    */
    function propose(
    address[] memory _targets,
    uint256[] memory _values,
    bytes[] memory _calldatas,
    string memory _title,
    string memory _description
    ) external returns (uint256) {
    require(
    getPriorVotes(msg.sender, block.number - 1) >= proposalThreshold,
    “Proposer votes below threshold”
    );
    require(
    _targets.length == _values.length && _targets.length == _calldatas.length,
    “Proposal function information mismatch”
    );
    require(_targets.length > 0, “Must provide actions”);
    require(_targets.length <= 10, “Too many actions”);
    
    proposalCount++;
    uint256 proposalId = proposalCount;
    
    Proposal storage newProposal = proposals[proposalId];
    newProposal.id = proposalId;
    newProposal.proposer = msg.sender;
    newProposal.title = _title;
    newProposal.description = _description;
    newProposal.targets = _targets;
    newProposal.values = _values;
    newProposal.calldatas = _calldatas;
    newProposal.startBlock = block.number + votingDelay;
    newProposal.endBlock = newProposal.startBlock + votingPeriod;
    
    emit ProposalCreated(
    proposalId,
    msg.sender,
    _title,
    newProposal.startBlock,
    newProposal.endBlock
    );
    
    return proposalId;
    }
  
  /**
  - @notice Cast a vote
    */
    function castVote(uint256 _proposalId, VoteType _support) external returns (uint256) {
    return _castVote(msg.sender, _proposalId, _support, “”);
    }
  
  /**
  - @notice Cast a vote with reason
    */
    function castVoteWithReason(
    uint256 _proposalId,
    VoteType _support,
    string calldata _reason
    ) external returns (uint256) {
    return _castVote(msg.sender, _proposalId, _support, _reason);
    }
  
  function _castVote(
  address _voter,
  uint256 _proposalId,
  VoteType _support,
  string memory _reason
  ) internal returns (uint256) {
  require(state(_proposalId) == ProposalState.Active, “Voting is closed”);
  
  ```
   Proposal storage proposal = proposals[_proposalId];
   Receipt storage receipt = receipts[_proposalId][_voter];
   
   require(!receipt.hasVoted, "Already voted");
   
   uint256 votes = getPriorVotes(_voter, proposal.startBlock);
   
   if (_support == VoteType.Against) {
       proposal.againstVotes += votes;
   } else if (_support == VoteType.For) {
       proposal.forVotes += votes;
   } else if (_support == VoteType.Abstain) {
       proposal.abstainVotes += votes;
   }
   
   receipt.hasVoted = true;
   receipt.support = _support;
   receipt.votes = votes;
   
   emit VoteCast(_voter, _proposalId, _support, votes, _reason);
   
   return votes;
  ```
  
  }
  
  /**
  - @notice Queue successful proposal
    */
    function queue(uint256 _proposalId) external {
    require(state(_proposalId) == ProposalState.Succeeded, “Proposal not succeeded”);
    
    Proposal storage proposal = proposals[_proposalId];
    uint256 eta = block.timestamp + timelockDelay;
    proposal.eta = eta;
    
    emit ProposalQueued(_proposalId, eta);
    }
  
  /**
  - @notice Execute queued proposal
    */
    function execute(uint256 _proposalId) external payable {
    require(state(_proposalId) == ProposalState.Queued, “Proposal not queued”);
    
    Proposal storage proposal = proposals[_proposalId];
    require(block.timestamp >= proposal.eta, “Timelock not met”);
    
    proposal.executed = true;
    
    for (uint256 i = 0; i < proposal.targets.length; i++) {
    (bool success, ) = proposal.targets[i].call{value: proposal.values[i]}(
    proposal.calldatas[i]
    );
    require(success, “Transaction execution reverted”);
    }
    
    emit ProposalExecuted(_proposalId);
    }
  
  /**
  - @notice Cancel a proposal
    */
    function cancel(uint256 _proposalId) external {
    require(state(_proposalId) != ProposalState.Executed, “Cannot cancel executed”);
    
    Proposal storage proposal = proposals[_proposalId];
    require(
    msg.sender == proposal.proposer ||
    getPriorVotes(proposal.proposer, block.number - 1) < proposalThreshold,
    “Proposer above threshold”
    );
    
    proposal.canceled = true;
    emit ProposalCanceled(_proposalId);
    }
  
  /**
  - @notice Get proposal state
    */
    function state(uint256 _proposalId) public view returns (ProposalState) {
    require(proposalCount >= _proposalId && _proposalId > 0, “Invalid proposal”);
    
    Proposal storage proposal = proposals[_proposalId];
    
    if (proposal.canceled) {
    return ProposalState.Canceled;
    } else if (block.number <= proposal.startBlock) {
    return ProposalState.Pending;
    } else if (block.number <= proposal.endBlock) {
    return ProposalState.Active;
    } else if (proposal.forVotes <= proposal.againstVotes || proposal.forVotes < quorumVotes) {
    return ProposalState.Defeated;
    } else if (proposal.eta == 0) {
    return ProposalState.Succeeded;
    } else if (proposal.executed) {
    return ProposalState.Executed;
    } else if (block.timestamp >= proposal.eta + 14 days) {
    return ProposalState.Expired;
    } else {
    return ProposalState.Queued;
    }
    }
  
  /**
  - @notice Get proposal details
    */
    function getProposal(uint256 _proposalId) external view returns (ProposalInfo memory) {
    Proposal storage proposal = proposals[_proposalId];
    
    return ProposalInfo({
    id: proposal.id,
    proposer: proposal.proposer,
    title: proposal.title,
    description: proposal.description,
    startBlock: proposal.startBlock,
    endBlock: proposal.endBlock,
    forVotes: proposal.forVotes,
    againstVotes: proposal.againstVotes,
    abstainVotes: proposal.abstainVotes,
    canceled: proposal.canceled,
    executed: proposal.executed,
    state: state(_proposalId)
    });
    }
  
  /**
  - @notice Get proposal actions
    */
    function getActions(uint256 _proposalId) external view returns (
    address[] memory targets,
    uint256[] memory values,
    bytes[] memory calldatas
    ) {
    Proposal storage proposal = proposals[_proposalId];
    return (proposal.targets, proposal.values, proposal.calldatas);
    }
  
  /**
  - @notice Get voter receipt
    */
    function getReceipt(uint256 _proposalId, address _voter) external view returns (
    bool hasVoted,
    VoteType support,
    uint256 votes
    ) {
    Receipt storage receipt = receipts[_proposalId][_voter];
    return (receipt.hasVoted, receipt.support, receipt.votes);
    }
  
  /**
  - @notice Get active proposals
    */
    function getActiveProposals() external view returns (uint256[] memory) {
    uint256[] memory activeIds = new uint256[](proposalCount);
    uint256 count = 0;
    
    for (uint256 i = 1; i <= proposalCount; i++) {
    if (state(i) == ProposalState.Active) {
    activeIds[count] = i;
    count++;
    }
    }
    
    uint256[] memory result = new uint256[](count);
    for (uint256 i = 0; i < count; i++) {
    result[i] = activeIds[i];
    }
    
    return result;
    }
  
  /**
  - @notice Check if can vote
    */
    function canVote(uint256 _proposalId, address _voter) external view returns (bool, uint256) {
    if (state(_proposalId) != ProposalState.Active) {
    return (false, 0);
    }
    
    Receipt storage receipt = receipts[_proposalId][_voter];
    if (receipt.hasVoted) {
    return (false, 0);
    }
    
    uint256 votes = getPriorVotes(_voter, proposals[_proposalId].startBlock);
    return (votes > 0, votes);
    }
  
  // ============================================
  // ADMIN FUNCTIONS
  // ============================================
  
  function setVotingDelay(uint256 _newDelay) external onlyAdmin {
  votingDelay = _newDelay;
  }
  
  function setVotingPeriod(uint256 _newPeriod) external onlyAdmin {
  votingPeriod = _newPeriod;
  }
  
  function setProposalThreshold(uint256 _newThreshold) external onlyAdmin {
  proposalThreshold = _newThreshold;
  }
  
  function setQuorumVotes(uint256 _newQuorum) external onlyAdmin {
  quorumVotes = _newQuorum;
  }
  
  function setTimelockDelay(uint256 _newDelay) external onlyAdmin {
  timelockDelay = _newDelay;
  }
  
  receive() external payable {}
  }