-- Schema for the network intrusion detection serving layer.
-- Spark (score_and_load.py) truncates and reloads these tables on every
-- pipeline run; the API only ever reads from them.
--
-- NSL-KDD and CICIDS2017 don't share raw feature columns (see spark_jobs/
-- common.py for why), so this table normalizes only the handful of fields
-- that mean roughly the same thing across both (duration, bytes sent/
-- received, protocol where available) and stores each row's full,
-- dataset-specific feature vector as JSONB - the same "normalize the common
-- bits, keep the raw event as JSON" pattern real security data pipelines use
-- for heterogeneous log sources.

CREATE TABLE IF NOT EXISTS flows_scored (
    flow_id                 VARCHAR(64) PRIMARY KEY,
    dataset_source           VARCHAR(16) NOT NULL,  -- 'nsl_kdd' | 'cicids2017'
    split                     VARCHAR(8) NOT NULL,   -- 'train' | 'test'
    source_day                VARCHAR(32),           -- CICIDS2017 only (e.g. 'friday_ddos'); null for NSL-KDD
    protocol                   VARCHAR(16),           -- NSL-KDD only (protocol_type); null for CICIDS2017 - see README
    duration                    DOUBLE PRECISION,
    bytes_sent                   DOUBLE PRECISION,
    bytes_received                 DOUBLE PRECISION,
    is_attack_actual                 INTEGER NOT NULL,
    attack_category_actual             VARCHAR(32) NOT NULL,
    predicted_label                       INTEGER NOT NULL,
    attack_probability                       DOUBLE PRECISION NOT NULL,
    rule_flags                                 VARCHAR(255),
    raw_features                                 JSONB NOT NULL,
    scored_at                                     TIMESTAMP NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_flows_dataset_source ON flows_scored (dataset_source);
CREATE INDEX IF NOT EXISTS idx_flows_split ON flows_scored (dataset_source, split);
CREATE INDEX IF NOT EXISTS idx_flows_predicted_label ON flows_scored (predicted_label);
CREATE INDEX IF NOT EXISTS idx_flows_category ON flows_scored (attack_category_actual);
CREATE INDEX IF NOT EXISTS idx_flows_raw_features ON flows_scored USING GIN (raw_features);

CREATE TABLE IF NOT EXISTS dataset_stats (
    dataset_source            VARCHAR(16) NOT NULL,
    split                      VARCHAR(8) NOT NULL,
    total_flows                 INTEGER NOT NULL,
    actual_attack_count           INTEGER NOT NULL,
    predicted_attack_count          INTEGER NOT NULL,
    avg_attack_probability             DOUBLE PRECISION,
    PRIMARY KEY (dataset_source, split)
);
