// SPDX-License-Identifier: MIT
pragma solidity ^0.8.19;

/**
 * @title ElectionVotingContract
 * @dev Secure, transparent, and auditable election voting system with fraud detection
 */
contract ElectionVotingContract {
    
    // ==================== STRUCTS ====================
    
    struct Vote {
        uint256 regionId;
        uint256 candidateId;
        bytes32 voterHash; // Hashed voter ID for privacy
        uint256 timestamp;
        bool isValid;
    }
    
    struct Region {
        string name;
        uint256 totalRegisteredVoters;
        uint256 totalVotesCast;
        bool isActive;
        mapping(uint256 => uint256) candidateVotes; // candidateId => vote count
        mapping(bytes32 => bool) hasVoted; // voterHash => voted status
    }
    
    struct Candidate {
        uint256 id;
        string name;
        string party;
        bool isActive;
    }
    
    struct AnomalyReport {
        uint256 regionId;
        string anomalyType;
        string description;
        uint256 timestamp;
        address reporter;
        bool isVerified;
    }
    
    // ==================== STATE VARIABLES ====================
    
    address public electionAuthority;
    uint256 public electionStartTime;
    uint256 public electionEndTime;
    bool public electionActive;
    
    mapping(uint256 => Region) public regions;
    mapping(uint256 => Candidate) public candidates;
    mapping(uint256 => Vote) public votes;
    mapping(uint256 => AnomalyReport) public anomalyReports;
    
    uint256 public regionCount;
    uint256 public candidateCount;
    uint256 public voteCount;
    uint256 public anomalyCount;
    
    mapping(address => bool) public authorizedValidators;
    mapping(address => bool) public authorizedObservers;
    
    // ==================== EVENTS ====================
    
    event ElectionInitialized(uint256 startTime, uint256 endTime);
    event RegionRegistered(uint256 indexed regionId, string name, uint256 registeredVoters);
    event CandidateRegistered(uint256 indexed candidateId, string name, string party);
    event VoteCast(uint256 indexed voteId, uint256 regionId, uint256 candidateId, uint256 timestamp);
    event AnomalyDetected(uint256 indexed anomalyId, uint256 regionId, string anomalyType);
    event AnomalyVerified(uint256 indexed anomalyId, bool isValid);
    event ElectionFinalized(uint256 timestamp, uint256 totalVotes);
    event ValidatorAdded(address indexed validator);
    event ObserverAdded(address indexed observer);
    
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
    
    // ==================== CONSTRUCTOR ====================
    
    constructor() {
        electionAuthority = msg.sender;
        authorizedValidators[msg.sender] = true;
        authorizedObservers[msg.sender] = true;
    }
    
    // ==================== ELECTION SETUP FUNCTIONS ====================
    
    function initializeElection(uint256 _startTime, uint256 _endTime) external onlyAuthority {
        require(_startTime > block.timestamp, "Start time must be in future");
        require(_endTime > _startTime, "End time must be after start");
        
        electionStartTime = _startTime;
        electionEndTime = _endTime;
        electionActive = true;
        
        emit ElectionInitialized(_startTime, _endTime);
    }
    
    function registerRegion(string memory _name, uint256 _registeredVoters) external onlyAuthority {
        regionCount++;
        Region storage newRegion = regions[regionCount];
        newRegion.name = _name;
        newRegion.totalRegisteredVoters = _registeredVoters;
        newRegion.isActive = true;
        
        emit RegionRegistered(regionCount, _name, _registeredVoters);
    }
    
    function registerCandidate(string memory _name, string memory _party) external onlyAuthority {
        candidateCount++;
        candidates[candidateCount] = Candidate({
            id: candidateCount,
            name: _name,
            party: _party,
            isActive: true
        });
        
        emit CandidateRegistered(candidateCount, _name, _party);
    }
    
    function addValidator(address _validator) external onlyAuthority {
        authorizedValidators[_validator] = true;
        emit ValidatorAdded(_validator);
    }
    
    function addObserver(address _observer) external onlyAuthority {
        authorizedObservers[_observer] = true;
        emit ObserverAdded(_observer);
    }
    
    // ==================== VOTING FUNCTIONS ====================
    
    function castVote(
        uint256 _regionId,
        uint256 _candidateId,
        bytes32 _voterHash
    ) external electionInProgress returns (uint256) {
        require(regions[_regionId].isActive, "Region not active");
        require(candidates[_candidateId].isActive, "Candidate not active");
        require(!regions[_regionId].hasVoted[_voterHash], "Already voted");
        
        // Check for overvoting anomaly
        if (regions[_regionId].totalVotesCast >= regions[_regionId].totalRegisteredVoters) {
            _reportAnomaly(_regionId, "OVERVOTE", "Vote count exceeds registered voters");
        }
        
        voteCount++;
        votes[voteCount] = Vote({
            regionId: _regionId,
            candidateId: _candidateId,
            voterHash: _voterHash,
            timestamp: block.timestamp,
            isValid: true
        });
        
        regions[_regionId].hasVoted[_voterHash] = true;
        regions[_regionId].candidateVotes[_candidateId]++;
        regions[_regionId].totalVotesCast++;
        
        emit VoteCast(voteCount, _regionId, _candidateId, block.timestamp);
        
        return voteCount;
    }
    
    function invalidateVote(uint256 _voteId) external onlyValidator {
        require(votes[_voteId].isValid, "Vote already invalid");
        votes[_voteId].isValid = false;
        
        uint256 regionId = votes[_voteId].regionId;
        uint256 candidateId = votes[_voteId].candidateId;
        
        regions[regionId].candidateVotes[candidateId]--;
        regions[regionId].totalVotesCast--;
    }
    
    // ==================== ANOMALY DETECTION FUNCTIONS ====================
    
    function reportAnomaly(
        uint256 _regionId,
        string memory _anomalyType,
        string memory _description
    ) external onlyObserver {
        _reportAnomaly(_regionId, _anomalyType, _description);
    }
    
    function _reportAnomaly(
        uint256 _regionId,
        string memory _anomalyType,
        string memory _description
    ) internal {
        anomalyCount++;
        anomalyReports[anomalyCount] = AnomalyReport({
            regionId: _regionId,
            anomalyType: _anomalyType,
            description: _description,
            timestamp: block.timestamp,
            reporter: msg.sender,
            isVerified: false
        });
        
        emit AnomalyDetected(anomalyCount, _regionId, _anomalyType);
    }
    
    function verifyAnomaly(uint256 _anomalyId, bool _isValid) external onlyValidator {
        require(_anomalyId <= anomalyCount, "Invalid anomaly ID");
        anomalyReports[_anomalyId].isVerified = _isValid;
        
        emit AnomalyVerified(_anomalyId, _isValid);
    }
    
    // ==================== QUERY FUNCTIONS ====================
    
    function getRegionResults(uint256 _regionId) external view returns (
        string memory name,
        uint256 totalVotes,
        uint256 registeredVoters,
        uint256 turnoutPercentage
    ) {
        Region storage region = regions[_regionId];
        name = region.name;
        totalVotes = region.totalVotesCast;
        registeredVoters = region.totalRegisteredVoters;
        turnoutPercentage = (totalVotes * 100) / registeredVoters;
    }
    
    function getCandidateVotes(uint256 _regionId, uint256 _candidateId) external view returns (uint256) {
        return regions[_regionId].candidateVotes[_candidateId];
    }
    
    function getTotalVotesForCandidate(uint256 _candidateId) external view returns (uint256 total) {
        for (uint256 i = 1; i <= regionCount; i++) {
            total += regions[i].candidateVotes[_candidateId];
        }
    }
    
    function hasVoted(uint256 _regionId, bytes32 _voterHash) external view returns (bool) {
        return regions[_regionId].hasVoted[_voterHash];
    }
    
    function getVoteDetails(uint256 _voteId) external view onlyObserver returns (
        uint256 regionId,
        uint256 candidateId,
        uint256 timestamp,
        bool isValid
    ) {
        Vote storage vote = votes[_voteId];
        return (vote.regionId, vote.candidateId, vote.timestamp, vote.isValid);
    }
    
    function getAnomalyReport(uint256 _anomalyId) external view returns (
        uint256 regionId,
        string memory anomalyType,
        string memory description,
        uint256 timestamp,
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
        uint256 startTime,
        uint256 endTime,
        uint256 totalVotes,
        uint256 totalAnomalies
    ) {
        return (
            electionActive,
            electionStartTime,
            electionEndTime,
            voteCount,
            anomalyCount
        );
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