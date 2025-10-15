// SPDX-License-Identifier: MIT
pragma solidity ^0.8.19;

/**
 * @title IdentityRegistry
 * @dev Central registry for managing verified identities with reputation
 */
contract IdentityRegistry {
    struct VerifiedIdentity {
        address owner;
        uint256 reputationScore;
        uint256 verificationsCount;
        mapping(address => bool) verifiedBy;
        mapping(string => Credential) credentials;
        bool isActive;
    }
    
    struct Credential {
        string credentialType;
        string issuer;
        uint256 issuedAt;
        uint256 expiresAt;
        bool isValid;
        bytes32 dataHash;
    }
    
    mapping(address => VerifiedIdentity) private identities;
    mapping(address => bool) public authorizedVerifiers;
    
    address public admin;
    uint256 public constant MIN_REPUTATION = 0;
    uint256 public constant MAX_REPUTATION = 1000;
    
    event IdentityRegistered(address indexed owner, uint256 timestamp);
    event ReputationUpdated(address indexed owner, uint256 newScore);
    event CredentialIssued(address indexed owner, string credentialType);
    event VerifierAuthorized(address indexed verifier);
    event IdentityDeactivated(address indexed owner);
    
    modifier onlyAdmin() {
        require(msg.sender == admin, "Only admin");
        _;
    }
    
    modifier onlyVerifier() {
        require(authorizedVerifiers[msg.sender], "Not authorized verifier");
        _;
    }
    
    constructor() {
        admin = msg.sender;
        authorizedVerifiers[msg.sender] = true;
    }
    
    function registerIdentity() external {
        require(!identities[msg.sender].isActive, "Already registered");
        
        VerifiedIdentity storage identity = identities[msg.sender];
        identity.owner = msg.sender;
        identity.reputationScore = 500; // Start with neutral reputation
        identity.isActive = true;
        
        emit IdentityRegistered(msg.sender, block.timestamp);
    }
    
    function issueCredential(
        address _owner,
        string memory _credentialType,
        string memory _issuer,
        uint256 _validityPeriod,
        bytes32 _dataHash
    ) external onlyVerifier {
        require(identities[_owner].isActive, "Identity not registered");
        
        Credential storage cred = identities[_owner].credentials[_credentialType];
        cred.credentialType = _credentialType;
        cred.issuer = _issuer;
        cred.issuedAt = block.timestamp;
        cred.expiresAt = block.timestamp + _validityPeriod;
        cred.isValid = true;
        cred.dataHash = _dataHash;
        
        identities[_owner].verificationsCount++;
        
        emit CredentialIssued(_owner, _credentialType);
    }
    
    function updateReputation(address _owner, int256 _change) external onlyVerifier {
        require(identities[_owner].isActive, "Identity not registered");
        
        VerifiedIdentity storage identity = identities[_owner];
        
        if (_change > 0) {
            identity.reputationScore = min(
                identity.reputationScore + uint256(_change),
                MAX_REPUTATION
            );
        } else {
            uint256 decrease = uint256(-_change);
            if (identity.reputationScore > decrease) {
                identity.reputationScore -= decrease;
            } else {
                identity.reputationScore = MIN_REPUTATION;
            }
        }
        
        emit ReputationUpdated(_owner, identity.reputationScore);
    }
    
    function verifyCredential(address _owner, string memory _credentialType) 
        external 
        view 
        returns (bool) 
    {
        Credential storage cred = identities[_owner].credentials[_credentialType];
        return cred.isValid && block.timestamp < cred.expiresAt;
    }
    
    function getReputation(address _owner) external view returns (uint256) {
        return identities[_owner].reputationScore;
    }
    
    function authorizeVerifier(address _verifier) external onlyAdmin {
        authorizedVerifiers[_verifier] = true;
        emit VerifierAuthorized(_verifier);
    }
    
    function deactivateIdentity(address _owner) external onlyAdmin {
        identities[_owner].isActive = false;
        emit IdentityDeactivated(_owner);
    }
    
    function min(uint256 a, uint256 b) private pure returns (uint256) {
        return a < b ? a : b;
    }
}

/**
 * @title IdentityGovernance
 * @dev Decentralized governance for identity protocol
 */
contract IdentityGovernance {
    struct Proposal {
        uint256 id;
        address proposer;
        string description;
        uint256 forVotes;
        uint256 againstVotes;
        uint256 startTime;
        uint256 endTime;
        bool executed;
        mapping(address => bool) hasVoted;
        ProposalType proposalType;
        bytes callData;
    }
    
    enum ProposalType {
        AddVerifier,
        RemoveVerifier,
        UpdateParameter,
        EmergencyStop
    }
    
    mapping(uint256 => Proposal) public proposals;
    mapping(address => uint256) public votingPower;
    
    uint256 public proposalCount;
    uint256 public constant VOTING_PERIOD = 3 days;
    uint256 public constant QUORUM = 1000; // Minimum votes needed
    
    address public identityContract;
    bool public paused;
    
    event ProposalCreated(uint256 indexed proposalId, address indexed proposer);
    event Voted(uint256 indexed proposalId, address indexed voter, bool support);
    event ProposalExecuted(uint256 indexed proposalId);
    event EmergencyStopActivated(address indexed activator);
    
    modifier notPaused() {
        require(!paused, "Contract is paused");
        _;
    }
    
    constructor(address _identityContract) {
        identityContract = _identityContract;
    }
    
    function createProposal(
        string memory _description,
        ProposalType _proposalType,
        bytes memory _callData
    ) external notPaused returns (uint256) {
        require(votingPower[msg.sender] > 0, "No voting power");
        
        proposalCount++;
        Proposal storage proposal = proposals[proposalCount];
        proposal.id = proposalCount;
        proposal.proposer = msg.sender;
        proposal.description = _description;
        proposal.startTime = block.timestamp;
        proposal.endTime = block.timestamp + VOTING_PERIOD;
        proposal.proposalType = _proposalType;
        proposal.callData = _callData;
        
        emit ProposalCreated(proposalCount, msg.sender);
        
        return proposalCount;
    }
    
    function vote(uint256 _proposalId, bool _support) external notPaused {
        Proposal storage proposal = proposals[_proposalId];
        require(block.timestamp < proposal.endTime, "Voting ended");
        require(!proposal.hasVoted[msg.sender], "Already voted");
        require(votingPower[msg.sender] > 0, "No voting power");
        
        proposal.hasVoted[msg.sender] = true;
        
        if (_support) {
            proposal.forVotes += votingPower[msg.sender];
        } else {
            proposal.againstVotes += votingPower[msg.sender];
        }
        
        emit Voted(_proposalId, msg.sender, _support);
    }
    
    function executeProposal(uint256 _proposalId) external notPaused {
        Proposal storage proposal = proposals[_proposalId];
        require(block.timestamp >= proposal.endTime, "Voting not ended");
        require(!proposal.executed, "Already executed");
        require(proposal.forVotes > proposal.againstVotes, "Proposal failed");
        require(proposal.forVotes >= QUORUM, "Quorum not reached");
        
        proposal.executed = true;
        
        if (proposal.proposalType == ProposalType.EmergencyStop) {
            paused = true;
            emit EmergencyStopActivated(msg.sender);
        }
        
        emit ProposalExecuted(_proposalId);
    }
    
    function delegateVotingPower(address _delegatee, uint256 _amount) external {
        require(votingPower[msg.sender] >= _amount, "Insufficient voting power");
        votingPower[msg.sender] -= _amount;
        votingPower[_delegatee] += _amount;
    }
    
    function getProposalStatus(uint256 _proposalId) 
        external 
        view 
        returns (
            uint256 forVotes,
            uint256 againstVotes,
            bool isActive,
            bool executed
        ) 
    {
        Proposal storage proposal = proposals[_proposalId];
        return (
            proposal.forVotes,
            proposal.againstVotes,
            block.timestamp < proposal.endTime,
            proposal.executed
        );
    }
}

/**
 * @title BiometricVerification
 * @dev Handles biometric verification hashes for identity security
 */
contract BiometricVerification {
    struct BiometricData {
        bytes32 dataHash;
        uint256 timestamp;
        bool isVerified;
        BiometricType biometricType;
    }
    
    enum BiometricType {
        Fingerprint,
        FaceID,
        IrisScanner,
        VoiceRecognition
    }
    
    mapping(address => mapping(BiometricType => BiometricData)) private biometrics;
    mapping(address => bool) public hasBiometrics;
    
    event BiometricRegistered(address indexed owner, BiometricType biometricType);
    event BiometricVerified(address indexed owner, BiometricType biometricType);
    event BiometricRemoved(address indexed owner, BiometricType biometricType);
    
    function registerBiometric(
        BiometricType _type,
        bytes32 _dataHash
    ) external {
        require(_dataHash != bytes32(0), "Invalid hash");
        
        biometrics[msg.sender][_type] = BiometricData({
            dataHash: _dataHash,
            timestamp: block.timestamp,
            isVerified: false,
            biometricType: _type
        });
        
        hasBiometrics[msg.sender] = true;
        
        emit BiometricRegistered(msg.sender, _type);
    }
    
    function verifyBiometric(
        address _owner,
        BiometricType _type,
        bytes32 _providedHash
    ) external view returns (bool) {
        BiometricData storage data = biometrics[_owner][_type];
        return data.dataHash == _providedHash && data.dataHash != bytes32(0);
    }
    
    function updateBiometric(
        BiometricType _type,
        bytes32 _newDataHash
    ) external {
        require(hasBiometrics[msg.sender], "No biometrics registered");
        
        biometrics[msg.sender][_type].dataHash = _newDataHash;
        biometrics[msg.sender][_type].timestamp = block.timestamp;
    }
    
    function removeBiometric(BiometricType _type) external {
        delete biometrics[msg.sender][_type];
        emit BiometricRemoved(msg.sender, _type);
    }
    
    function getBiometricTimestamp(address _owner, BiometricType _type) 
        external 
        view 
        returns (uint256) 
    {
        return biometrics[_owner][_type].timestamp;
    }
}

/**
 * @title RateLimiter
 * @dev Implements rate limiting for sensitive operations
 */
contract RateLimiter {
    struct RateLimit {
        uint256 requestCount;
        uint256 windowStart;
    }
    
    mapping(address => mapping(bytes4 => RateLimit)) private limits;
    
    uint256 public constant RATE_LIMIT_WINDOW = 1 hours;
    uint256 public constant MAX_REQUESTS = 10;
    
    event RateLimitExceeded(address indexed user, bytes4 indexed functionSig);
    
    modifier rateLimit() {
        bytes4 sig = msg.sig;
        RateLimit storage limit = limits[msg.sender][sig];
        
        if (block.timestamp > limit.windowStart + RATE_LIMIT_WINDOW) {
            limit.requestCount = 0;
            limit.windowStart = block.timestamp;
        }
        
        require(limit.requestCount < MAX_REQUESTS, "Rate limit exceeded");
        limit.requestCount++;
        
        _;
    }
    
    function getRateLimit(address _user, bytes4 _functionSig) 
        external 
        view 
        returns (uint256 requestCount, uint256 windowStart) 
    {
        RateLimit storage limit = limits[_user][_functionSig];
        return (limit.requestCount, limit.windowStart);
    }
}