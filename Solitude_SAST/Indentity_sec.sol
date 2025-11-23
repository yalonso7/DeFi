// SPDX-License-Identifier: MIT
pragma solidity ^0.8.19;

/**
 * @title DecentralizedIdentity
 * @dev Manages secure digital identities on blockchain
 */
contract DecentralizedIdentity {
    
    struct Identity {
        address owner;
        string name;
        string email;
        bool isVerified;
        uint256 createdAt;
        uint256 updatedAt;
        mapping(string => string) attributes;
        mapping(address => bool) authorizedAccess;
    }
    
    struct AccessRequest {
        address requester;
        uint256 timestamp;
        bool approved;
        string purpose;
    }
    
    mapping(address => Identity) private identities;
    mapping(address => bool) public hasIdentity;
    mapping(address => AccessRequest[]) public accessRequests;
    mapping(address => address[]) private verifiers;
    
    address public admin;
    uint256 public totalIdentities;
    
    event IdentityCreated(address indexed owner, uint256 timestamp);
    event IdentityUpdated(address indexed owner, uint256 timestamp);
    event IdentityVerified(address indexed owner, address indexed verifier);
    event AccessGranted(address indexed owner, address indexed accessor);
    event AccessRevoked(address indexed owner, address indexed accessor);
    event AccessRequested(address indexed requester, address indexed owner);
    event AttributeAdded(address indexed owner, string key);
    
    modifier onlyIdentityOwner() {
        require(hasIdentity[msg.sender], "No identity found");
        require(identities[msg.sender].owner == msg.sender, "Not identity owner");
        _;
    }
    
    modifier identityExists(address _address) {
        require(hasIdentity[_address], "Identity does not exist");
        _;
    }
    
    constructor() {
        admin = msg.sender;
    }
    
    /**
     * @dev Create a new identity
     */
    function createIdentity(string memory _name, string memory _email) external {
        require(!hasIdentity[msg.sender], "Identity already exists");
        require(bytes(_name).length > 0, "Name required");
        
        Identity storage newIdentity = identities[msg.sender];
        newIdentity.owner = msg.sender;
        newIdentity.name = _name;
        newIdentity.email = _email;
        newIdentity.isVerified = false;
        newIdentity.createdAt = block.timestamp;
        newIdentity.updatedAt = block.timestamp;
        
        hasIdentity[msg.sender] = true;
        totalIdentities++;
        
        emit IdentityCreated(msg.sender, block.timestamp);
    }
    
    /**
     * @dev Update identity information
     */
    function updateIdentity(string memory _name, string memory _email) 
        external 
        onlyIdentityOwner 
    {
        Identity storage identity = identities[msg.sender];
        identity.name = _name;
        identity.email = _email;
        identity.updatedAt = block.timestamp;
        
        emit IdentityUpdated(msg.sender, block.timestamp);
    }
    
    /**
     * @dev Add custom attribute to identity
     */
    function addAttribute(string memory _key, string memory _value) 
        external 
        onlyIdentityOwner 
    {
        identities[msg.sender].attributes[_key] = _value;
        identities[msg.sender].updatedAt = block.timestamp;
        
        emit AttributeAdded(msg.sender, _key);
    }
    
    /**
     * @dev Get attribute value
     */
    function getAttribute(address _owner, string memory _key) 
        external 
        view 
        identityExists(_owner)
        returns (string memory) 
    {
        require(
            msg.sender == _owner || 
            identities[_owner].authorizedAccess[msg.sender] ||
            msg.sender == admin,
            "Access denied"
        );
        return identities[_owner].attributes[_key];
    }
    
    /**
     * @dev Verify an identity (admin or authorized verifier only)
     */
    function verifyIdentity(address _identity) external {
        require(msg.sender == admin, "Only admin can verify");
        require(hasIdentity[_identity], "Identity does not exist");
        
        identities[_identity].isVerified = true;
        verifiers[_identity].push(msg.sender);
        
        emit IdentityVerified(_identity, msg.sender);
    }
    
    /**
     * @dev Request access to someone's identity
     */
    function requestAccess(address _owner, string memory _purpose) 
        external 
        identityExists(_owner) 
    {
        require(msg.sender != _owner, "Cannot request access to own identity");
        
        AccessRequest memory newRequest = AccessRequest({
            requester: msg.sender,
            timestamp: block.timestamp,
            approved: false,
            purpose: _purpose
        });
        
        accessRequests[_owner].push(newRequest);
        
        emit AccessRequested(msg.sender, _owner);
    }
    
    /**
     * @dev Grant access to requester
     */
    function grantAccess(address _accessor) external onlyIdentityOwner {
        identities[msg.sender].authorizedAccess[_accessor] = true;
        emit AccessGranted(msg.sender, _accessor);
    }
    
    /**
     * @dev Revoke access from accessor
     */
    function revokeAccess(address _accessor) external onlyIdentityOwner {
        identities[msg.sender].authorizedAccess[_accessor] = false;
        emit AccessRevoked(msg.sender, _accessor);
    }
    
    /**
     * @dev Check if address has access to identity
     */
    function hasAccess(address _owner, address _accessor) 
        external 
        view 
        identityExists(_owner)
        returns (bool) 
    {
        return identities[_owner].authorizedAccess[_accessor];
    }
    
    /**
     * @dev Get identity information
     */
    function getIdentity(address _owner) 
        external 
        view 
        identityExists(_owner)
        returns (
            address owner,
            string memory name,
            string memory email,
            bool isVerified,
            uint256 createdAt,
            uint256 updatedAt
        ) 
    {
        require(
            msg.sender == _owner || 
            identities[_owner].authorizedAccess[msg.sender] ||
            msg.sender == admin,
            "Access denied"
        );
        
        Identity storage identity = identities[_owner];
        return (
            identity.owner,
            identity.name,
            identity.email,
            identity.isVerified,
            identity.createdAt,
            identity.updatedAt
        );
    }
    
    /**
     * @dev Get all access requests for caller's identity
     */
    function getAccessRequests() 
        external 
        view 
        onlyIdentityOwner
        returns (AccessRequest[] memory) 
    {
        return accessRequests[msg.sender];
    }
    
    /**
     * @dev Check if identity is verified
     */
    function isVerified(address _owner) 
        external 
        view 
        identityExists(_owner)
        returns (bool) 
    {
        return identities[_owner].isVerified;
    }
}