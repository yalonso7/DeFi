// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

contract CollateralGuard {
    struct CollateralSnapshot {
        uint256 timestamp;
        uint256 collateralAmount;
        uint256 debtAmount;
        address token;
    }
    
    mapping(address => CollateralSnapshot[]) public userSnapshots;
    uint256 public constant SNAPSHOT_WINDOW = 5; // Number of snapshots to maintain
    uint256 public constant MIN_TIME_BETWEEN_SNAPSHOTS = 1 minutes;
    uint256 public constant MAX_DEBT_TO_COLLATERAL_RATIO = 90; // 90%
    
    event AnomalyDetected(
        address user,
        uint256 currentRatio,
        uint256 averageRatio
    );
    
    function recordCollateralSnapshot(
        address user,
        uint256 collateralAmount,
        uint256 debtAmount,
        address token
    ) external {
        require(collateralAmount > 0, "Invalid collateral amount");
        
        // Create new snapshot
        CollateralSnapshot memory newSnapshot = CollateralSnapshot({
            timestamp: block.timestamp,
            collateralAmount: collateralAmount,
            debtAmount: debtAmount,
            token: token
        });
        
        // Add to user's snapshots
        if (userSnapshots[user].length >= SNAPSHOT_WINDOW) {
            // Remove oldest snapshot
            for (uint i = 0; i < SNAPSHOT_WINDOW - 1; i++) {
                userSnapshots[user][i] = userSnapshots[user][i + 1];
            }
            userSnapshots[user][SNAPSHOT_WINDOW - 1] = newSnapshot;
        } else {
            userSnapshots[user].push(newSnapshot);
        }
        
        // Check for anomalies
        checkForAnomalies(user);
    }
    
    function checkForAnomalies(address user) internal {
        if (userSnapshots[user].length < 2) return;
        
        uint256 currentRatio = calculateRatio(
            userSnapshots[user][userSnapshots[user].length - 1].debtAmount,
            userSnapshots[user][userSnapshots[user].length - 1].collateralAmount
        );
        
        uint256 averageRatio = calculateAverageRatio(user);
        
        // Detect sudden changes in debt-to-collateral ratio
        if (currentRatio > averageRatio * 150 / 100 || // 50% increase
            currentRatio > MAX_DEBT_TO_COLLATERAL_RATIO) {
            emit AnomalyDetected(user, currentRatio, averageRatio);
            // Additional protective actions can be implemented here
        }
    }
    
    function calculateRatio(uint256 debt, uint256 collateral) internal pure returns (uint256) {
        if (collateral == 0) return 0;
        return (debt * 100) / collateral;
    }
    
    function calculateAverageRatio(address user) internal view returns (uint256) {
        uint256 total = 0;
        uint256 count = userSnapshots[user].length;
        
        for (uint i = 0; i < count; i++) {
            total += calculateRatio(
                userSnapshots[user][i].debtAmount,
                userSnapshots[user][i].collateralAmount
            );
        }
        
        return total / count;
    }
}