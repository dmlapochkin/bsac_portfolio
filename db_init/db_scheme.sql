CREATE TABLE securities (
    ticker VARCHAR(20) PRIMARY KEY,
    name VARCHAR(150),
    short_name VARCHAR(50),
    sector varchar(100),
    color VARCHAR(7),
    board VARCHAR(10),
    market VARCHAR(20),
    instrument_type VARCHAR(10),
    security_type VARCHAR(10),
    currency VARCHAR(3),
    lot_size INTEGER DEFAULT 1,
    min_step DECIMAL(20, 10),
    decimals SMALLINT DEFAULT 2,
    isin VARCHAR(12) UNIQUE,
    issue_size BIGINT,
    listing_level SMALLINT,
    is_active BOOLEAN,
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE indices (
    index_id VARCHAR(20) PRIMARY KEY,
    name VARCHAR(150) NOT NULL,
    short_name VARCHAR(100),
    color VARCHAR(7),
    board VARCHAR(10),
    currency VARCHAR(3),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE market_data (
    date DATE NOT NULL,
    ticker VARCHAR(20) REFERENCES securities(ticker) ON DELETE CASCADE,
    open DECIMAL(20, 10),
    high DECIMAL(20, 10),
    low DECIMAL(20, 10),
    close DECIMAL(20, 10),
    volume BIGINT,
    value DECIMAL(30, 4),
    PRIMARY KEY (date, ticker)
);

CREATE TABLE index_composition (
    index_id VARCHAR(20) REFERENCES indices(index_id) ON DELETE CASCADE,
    date DATE NOT NULL,
    ticker VARCHAR(20) REFERENCES securities(ticker) ON DELETE CASCADE,
    weight DECIMAL(10, 8) NOT NULL,
    PRIMARY KEY (index_id, date, ticker)
);

CREATE TABLE index_data (
    date DATE NOT NULL,
    index_id VARCHAR(20) REFERENCES indices(index_id) ON DELETE CASCADE,
    open DECIMAL(20, 10),
    high DECIMAL(20, 10),
    low DECIMAL(20, 10),
    close DECIMAL(20, 10),
    volume BIGINT,
    value DECIMAL(30, 4),
    PRIMARY KEY (date, index_id)
);

CREATE TABLE risk_profiles (
    profile_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    display_name VARCHAR(50) UNIQUE NOT NULL,
    description TEXT,
    is_active BOOLEAN,
    rebalance_policy_planned INT,
    rebalance_policy_benchmark NUMERIC(3,2),
    rebalance_policy_weights NUMERIC(3,2),
    icon VARCHAR(50),
    model_name VARCHAR(50),
    eta FLOAT
);

CREATE TABLE sessions (
    session_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    profile_id UUID REFERENCES risk_profiles(profile_id) ON DELETE SET NULL,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    is_active BOOLEAN DEFAULT true,
    current_balance DECIMAL(15,2) NOT NULL,
    rebalance_policy_planned INT,
    rebalance_policy_benchmark NUMERIC(3,2),
    rebalance_policy_weights NUMERIC(3,2)
);

CREATE TABLE portfolios (
    portfolio_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    session_id UUID REFERENCES sessions(session_id) ON DELETE CASCADE,
    initial_balance DECIMAL(15,2) NOT NULL,
    current_balance DECIMAL(15,2) NOT NULL,
    status VARCHAR(20) DEFAULT 'active',
    last_z FLOAT DEFAULT 1.0,
    horizon_days INT,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    closed_at TIMESTAMPTZ
);

CREATE TABLE portfolio_daily_snapshots (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    portfolio_id UUID REFERENCES portfolios(portfolio_id) ON DELETE CASCADE,
    snapshot_date DATE NOT NULL,
    cash NUMERIC(18, 4) NOT NULL DEFAULT 0.0,
    portfolio_value NUMERIC(18, 4) NOT NULL DEFAULT 0.0,
    z_benchmark NUMERIC(10, 6) NOT NULL DEFAULT 1.0,
    UNIQUE(portfolio_id, snapshot_date)
);

CREATE TABLE daily_portfolio_holdings (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    snapshot_id UUID REFERENCES portfolio_daily_snapshots(id) ON DELETE CASCADE,
    ticker VARCHAR(20) REFERENCES securities(ticker),
    lots INT NOT NULL,
    UNIQUE(snapshot_id, ticker)
);

CREATE TABLE portfolio_rebalances (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    portfolio_id UUID REFERENCES portfolios(portfolio_id),
    rebalance_date DATE NOT NULL,
    status VARCHAR(20) DEFAULT 'pending',
    reason TEXT,
    cash NUMERIC(15, 2),
    commission NUMERIC(15, 4) DEFAULT 0.0,
    CONSTRAINT uq_portfolio_rebalance_date UNIQUE (portfolio_id, rebalance_date)
);

CREATE TABLE rebalance_theoretical_weights (
    rebalance_id UUID NOT NULL REFERENCES portfolio_rebalances(id) ON DELETE CASCADE,
    ticker       VARCHAR(20) NOT NULL REFERENCES securities(ticker) ON DELETE CASCADE,
    weight       DECIMAL(10, 8) NOT NULL CHECK (weight >= 0.0 AND weight <= 1.0),
    PRIMARY KEY (rebalance_id, ticker)
);