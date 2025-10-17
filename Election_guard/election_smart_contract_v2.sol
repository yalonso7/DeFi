// SPDX-License-Identifier: MIT
pragma solidity ^0.8.19;

/**
 * @title ElectionVotingContract
 * @dev Secure, transparent, and auditable election voting system with fraud detection
 * @notice Optimized for gas efficiency and security
 */
contract ElectionVotingContract {
    
    // ==================== ENUMS ====================
    
    // Gas optimization: Using enum instead of strings for anomaly types
    enum AnomalyType {
        NONE,
        TURNOUT_SPIKE,
        OVERVOTE,
        BENFORD_VIOLATION,
        TIMESTAMP_MANIPULATION,
        DUPLICATE_VOTES,
        INVALID_RATIO
    }
    
    // ==================== STRUCTS ====================
    
    // Gas optimization: Using smaller uint types where possible
    struct Vote {
        uint32 regionId;
        uint32 candidateId;
        bytes32 voterHash; // Hashed voter ID for privacy
        uint64 timestamp;
        bool isValid;
    }
    
    struct Region {
        string name;
        uint32 totalRegisteredVoters;
        uint32 totalVotesCast;
        bool isActive;
        mapping(uint32 => uint32) candidateVotes; // candidateId => vote count
        mapping(bytes32 => bool) hasVoted; // voterHash => voted status
    }
    
    struct Candidate {
        uint32 id;
        string name;
        string party;
        bool isActive;
    }
    
    struct AnomalyReport {
        uint32 regionId;
        AnomalyType anomalyType; // Gas optimization: Using enum instead of string
        string description;
        uint64 timestamp;
        address reporter;
        bool isVerified;
    }
    
    // ==================== STATE VARIABLES ====================
    
    address public electionAuthority;
    uint64 public electionStartTime;
    uint64 public electionEndTime;
    bool public electionActive;
    bool private _paused; // Security enhancement: Emergency pause functionality
    
    mapping(uint32 => Region) public regions;
    mapping(uint32 => Candidate) public candidates;
    mapping(uint32 => Vote) public votes;
    mapping(uint32 => AnomalyReport) public anomalyReports;
    
    uint32 public regionCount;
    uint32 public candidateCount;
    uint32 public voteCount;
    uint32 public anomalyCount;
    
    mapping(address => bool) public authorizedValidators;
    mapping(address => bool) public authorizedObservers;
    
    // Security enhancement: Reentrancy guard
    uint256 private _reentrancyGuard;
    
    // ==================== EVENTS ====================
    
    event ElectionInitialized(uint64 startTime, uint64 endTime);
    event RegionRegistered(uint32 indexed regionId, string name, uint32 registeredVoters);
    event CandidateRegistered(uint32 indexed candidateId, string name, string party);
    event VoteCast(uint32 indexed voteId, uint32 regionId, uint32 candidateId, uint64 timestamp);
    event AnomalyDetected(uint32 indexed anomalyId, uint32 regionId, AnomalyType anomalyType);
    event AnomalyVerified(uint32 indexed anomalyId, bool isValid);
    event ElectionFinalized(uint64 timestamp, uint32 totalVotes);
    event ValidatorAdded(address indexed validator);
    event ObserverAdded(address indexed observer);
    event EmergencyPause(bool paused);
    
    // ==================== MODIFIERS ====================
    
    modifier onlyAuthority() {
        require(msg.sender == electionAuthority, "Only election authority");
        _;
    }
    
    modifier onlyValidator() {
        require(authorizedValidators[msg.sender] || msg.sender == electionAuthority, "Not authorized validator");
        _;
    }
    
    modifier onlyObserver() {
        require(authorizedObservers[msg.sender] || msg.sender == electionAuthority, "Not authorized observer");
        _;
    }
    
    modifier electionInProgress() {
        require(electionActive, "Election not active");
        require(block.timestamp >= electionStartTime, "Election not started");
        require(block.timestamp <= electionEndTime, "Election ended");
        _;
    }
    
    // Security enhancement: Reentrancy guard modifier
    modifier nonReentrant() {
        require(_reentrancyGuard == 0, "Reentrant call");
        _reentrancyGuard = 1;
        _;
        _reentrancyGuard = 0;
    }
    
    // Security enhancement: Emergency pause modifier
    modifier whenNotPaused() {
        require(!_paused, "Contract is paused");
        _;
    }
    
    // ==================== CONSTRUCTOR ====================
    
    constructor() {
        electionAuthority = msg.sender;
        authorizedValidators[msg.sender] = true;
        authorizedObservers[msg.sender] = true;
        _reentrancyGuard = 0;
        _paused = false;
    }
    
    // ==================== EMERGENCY CONTROLS ====================
    
    // Security enhancement: Emergency pause functionality
    function toggleEmergencyPause() external onlyAuthority {
        _paused = !_paused;
        emit EmergencyPause(_paused);
    }
    
    // ==================== ELECTION SETUP FUNCTIONS ====================
    
    function initializeElection(uint64 _startTime, uint64 _endTime) external onlyAuthority whenNotPaused {
        require(_startTime > uint64(block.timestamp), "Start time must be in future");
        require(_endTime > _startTime, "End time must be after start");
        
        electionStartTime = _startTime;
        electionEndTime = _endTime;
        electionActive = true;
        
        emit ElectionInitialized(_startTime, _endTime);
    }
    
    function registerRegion(string calldata _name, uint32 _registeredVoters) external onlyAuthority whenNotPaused {
        regionCount++;
        Region storage newRegion = regions[regionCount];
        newRegion.name = _name;
        newRegion.totalRegisteredVoters = _registeredVoters;
        newRegion.isActive = true;
        
        emit RegionRegistered(regionCount, _name, _registeredVoters);
    }
    
    function registerCandidate(string calldata _name, string calldata _party) external onlyAuthority whenNotPaused {
        candidateCount++;
        candidates[candidateCount] = Candidate({
            id: candidateCount,
            name: _name,
            party: _party,
            isActive: true
        });
        
        emit CandidateRegistered(candidateCount, _name, _party);
    }
    
    function addValidator(address _validator) external onlyAuthority whenNotPaused {
        require(_validator != address(0), "Invalid address");
        authorizedValidators[_validator] = true;
        emit ValidatorAdded(_validator);
    }
    
    function addObserver(address _observer) external onlyAuthority whenNotPaused {
        require(_observer != address(0), "Invalid address");
        authorizedObservers[_observer] = true;
        emit ObserverAdded(_observer);
    }
    
    // ==================== VOTING FUNCTIONS ====================
    
    // Security enhancement: Added nonReentrant modifier
    function castVote(
        uint32 _regionId,
        uint32 _candidateId,
        bytes32 _voterHash
    ) external electionInProgress nonReentrant whenNotPaused returns (uint32) {
        require(regions[_regionId].isActive, "Region not active");
        require(candidates[_candidateId].isActive, "Candidate not active");
        require(!regions[_regionId].hasVoted[_voterHash], "Already voted");
        
        // Check for overvoting anomaly
        if (regions[_regionId].totalVotesCast >= regions[_regionId].totalRegisteredVoters) {
            _reportAnomaly(_regionId, AnomalyType.OVERVOTE, "Vote count exceeds registered voters");
        }
        
        voteCount++;
        votes[voteCount] = Vote({
            regionId: _regionId,
            candidateId: _candidateId,
            voterHash: _voterHash,
            timestamp: uint64(block.timestamp),
            isValid: true
        });
        
        regions[_regionId].hasVoted[_voterHash] = true;
        regions[_regionId].candidateVotes[_candidateId]++;
        regions[_regionId].totalVotesCast++;
        
        emit VoteCast(voteCount, _regionId, _candidateId, uint64(block.timestamp));
        
        return voteCount;
    }
    
    // Batch voting for gas optimization
    function batchCastVotes(
        uint32[] calldata _regionIds,
        uint32[] calldata _candidateIds,
        bytes32[] calldata _voterHashes
    ) external electionInProgress nonReentrant whenNotPaused returns (uint32[] memory) {
        require(_regionIds.length == _candidateIds.length && _regionIds.length == _voterHashes.length, "Array length mismatch");
        require(_regionIds.length <= 10, "Batch too large"); // Prevent DOS attacks
        
        uint32[] memory voteIds = new uint32[](_regionIds.length);
        
        for (uint32 i = 0; i < _regionIds.length; i++) {
            uint32 _regionId = _regionIds[i];
            uint32 _candidateId = _candidateIds[i];
            bytes32 _voterHash = _voterHashes[i];
            
            require(regions[_regionId].isActive, "Region not active");
            require(candidates[_candidateId].isActive, "Candidate not active");
            require(!regions[_regionId].hasVoted[_voterHash], "Already voted");
            
            voteCount++;
            votes[voteCount] = Vote({
                regionId: _regionId,
                candidateId: _candidateId,
                voterHash: _voterHash,
                timestamp: uint64(block.timestamp),
                isValid: true
            });
            
            regions[_regionId].hasVoted[_voterHash] = true;
            regions[_regionId].candidateVotes[_candidateId]++;
            regions[_regionId].totalVotesCast++;
            
            emit VoteCast(voteCount, _regionId, _candidateId, uint64(block.timestamp));
            
            voteIds[i] = voteCount;
        }
        
        return voteIds;
    }
    
    function invalidateVote(uint32 _voteId) external onlyValidator nonReentrant whenNotPaused {
        require(votes[_voteId].isValid, "Vote already invalid");
        votes[_voteId].isValid = false;
        
        uint32 regionId = votes[_voteId].regionId;
        uint32 candidateId = votes[_voteId].candidateId;
        
        regions[regionId].candidateVotes[candidateId]--;
        regions[regionId].totalVotesCast--;
    }
    
    // ==================== ANOMALY DETECTION FUNCTIONS ====================
    
    function reportAnomaly(
        uint32 _regionId,
        AnomalyType _anomalyType,
        string calldata _description
    ) external onlyObserver whenNotPaused {
        _reportAnomaly(_regionId, _anomalyType, _description);
    }
    
    function _reportAnomaly(
        uint32 _regionId,
        AnomalyType _anomalyType,
        string memory _description
    ) internal {
        anomalyCount++;
        anomalyReports[anomalyCount] = AnomalyReport({
            regionId: _regionId,
            anomalyType: _anomalyType,
            description: _description,
            timestamp: uint64(block.timestamp),
            reporter: msg.sender,
            isVerified: false
        });
        
        emit AnomalyDetected(anomalyCount, _regionId, _anomalyType);
    }
    
    function verifyAnomaly(uint32 _anomalyId, bool _isValid) external onlyValidator whenNotPaused {
        require(_anomalyId <= anomalyCount, "Invalid anomaly ID");
        anomalyReports[_anomalyId].isVerified = _isValid;
        
        emit AnomalyVerified(_anomalyId, _isValid);
    }
    
    // ==================== QUERY FUNCTIONS ====================
    
    function getRegionResults(uint32 _regionId) external view returns (
        string memory name,
        uint32 totalVotes,
        uint32 registeredVoters,
        uint32 turnoutPercentage
    ) {
        Region storage region = regions[_regionId];
        name = region.name;
        totalVotes = region.totalVotesCast;
        registeredVoters = region.totalRegisteredVoters;
        // Prevent division by zero
        turnoutPercentage = registeredVoters > 0 ? (totalVotes * 100) / registeredVoters : 0;
    }
    
    function getCandidateVotes(uint32 _regionId, uint32 _candidateId) external view returns (uint32) {
        return regions[_regionId].candidateVotes[_candidateId];
    }
    
    function getTotalVotesForCandidate(uint32 _candidateId) external view returns (uint32 total) {
        for (uint32 i = 1; i <= regionCount; i++) {
            total += regions[i].candidateVotes[_candidateId];
        }
    }
    
    function hasVoted(uint32 _regionId, bytes32 _voterHash) external view returns (bool) {
        return regions[_regionId].hasVoted[_voterHash];
    }
    
    function getVoteDetails(uint32 _voteId) external view onlyObserver returns (
        uint32 regionId,
        uint32 candidateId,
        uint64 timestamp,
        bool isValid
    ) {
        Vote storage vote = votes[_voteId];
        return (vote.regionId, vote.candidateId, vote.timestamp, vote.isValid);
    }
    
    function getAnomalyReport(uint32 _anomalyId) external view returns (
        uint32 regionId,
        AnomalyType anomalyType,
        string memory description,
        uint64 timestamp,
        bool isVerified
    ) {
        AnomalyReport storage report = anomalyReports[_anomalyId];
        return (
            report.regionId,
            report.anomalyType,
            report.description,
            report.timestamp,
            report.isVerified
        );
    }
    
    function getElectionStatus() external view returns (
        bool isActive,
        uint64 startTime,
        uint64 endTime,
        uint32 totalVotes,
        uint32 totalAnomalies
    ) {
        return (
            electionActive,
            electionStartTime,
            electionEndTime,
            voteCount,
            anomalyCount
        );
    }
    
    // Emergency functions
    function toggleEmergencyPause() external onlyAuthority {
        _paused = !_paused;
        emit EmergencyPauseToggled(_paused);
    }
}
    
    // ==================== ELECTION FINALIZATION ====================
    
    function finalizeElection() external onlyAuthority {
        require(block.timestamp > electionEndTime, "Election still in progress");
        electionActive = false;
        
        emit ElectionFinalized(block.timestamp, voteCount);
    }
    
    // ==================== INTEGRITY VERIFICATION ====================
    
    function verifyRegionIntegrity(uint256 _regionId) external view returns (
        bool isValid,
        string memory message
    ) {
        Region storage region = regions[_regionId];
        
        // Check for overvoting
        if (region.totalVotesCast > region.totalRegisteredVoters) {
            return (false, "Overvoting detected: votes exceed registered voters");
        }
        
        // Check for suspicious turnout (>98%)
        uint256 turnout = (region.totalVotesCast * 100) / region.totalRegisteredVoters;
        if (turnout > 98) {
            return (false, "Suspicious turnout rate detected");
        }
        
        return (true, "Region integrity verified");
    }
    
    function getBlockchainHash() external view returns (bytes32) {
        return keccak256(abi.encodePacked(
            electionStartTime,
            electionEndTime,
            voteCount,
            block.timestamp
        ));
    }
}