# Key Features

Multi-Protocol Support:

- Add/remove lending protocols dynamically
- Tracks allocation across all protocols
- Activates/deactivates protocols as needed

Automatic Rebalancing:

- Monitors APY across all protocols
- Moves funds when APY difference exceeds threshold (default 0.5%)
- Time-gated to prevent excessive gas costs
- Emits events for tracking

User Management:

- Share-based accounting (like vault tokens)
- Proportional withdrawals
- Tracks individual user deposits
- Real-time value calculation

Fee System:

- Configurable performance fee (max 20%)
- Fee accumulation and collection
- Owner-controlled fee adjustments

Safety Features:

- Owner-only administrative functions
- Emergency withdrawal capability
- Validation checks on all operations
- Configurable thresholds

# How It Works

1. Deposit: Users deposit tokens, receive shares based on current pool value
1. Optimization: Funds automatically go to the highest-APY protocol
1. Rebalancing: Periodically checks all protocols and moves funds to maximize returns
1. Withdrawal: Users can withdraw anytime, getting their share of total value

# Configuration Options

- `minRebalanceThreshold`: Minimum APY difference to trigger rebalancing
- `rebalanceInterval`: Time between allowed rebalances
- `performanceFee`: Fee taken from profits

You’ll need to implement the `ILendingProtocol` interface for each protocol you want to integrate (like Aave, Compound, etc.).​​​​​​​​​​​​​​​​

