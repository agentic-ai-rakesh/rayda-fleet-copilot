-- Fleet telemetry database. Rebuilt from the raw dataset by `fleet-copilot ingest`.
-- Every table carries company_id (directly or through its snapshot), so the
-- tool layer can enforce tenant isolation with a single WHERE clause.

PRAGMA foreign_keys = ON;

CREATE TABLE companies (
    company_id    TEXT PRIMARY KEY,
    company_name  TEXT NOT NULL
);

CREATE TABLE employees (
    employee_id  TEXT PRIMARY KEY,
    company_id   TEXT NOT NULL REFERENCES companies (company_id)
);

CREATE TABLE devices (
    device_id         TEXT PRIMARY KEY,
    company_id        TEXT NOT NULL REFERENCES companies (company_id),
    employee_id       TEXT REFERENCES employees (employee_id),
    device_type       TEXT NOT NULL CHECK (device_type IN ('laptop', 'desktop')),
    serial_number     TEXT,
    model_name        TEXT NOT NULL,
    model_identifier  TEXT,
    processor         TEXT,
    memory_total_bytes INTEGER NOT NULL,
    os_platform       TEXT NOT NULL,
    os_name           TEXT NOT NULL,
    first_seen_at     TEXT NOT NULL,
    last_seen_at      TEXT NOT NULL,
    snapshot_count    INTEGER NOT NULL
);

-- One row per device per collection. snapshot_id is the evidence handle the
-- agent cites, e.g. "1LYSSFD074BB@20260514T0954Z".
CREATE TABLE snapshots (
    snapshot_id            TEXT PRIMARY KEY,
    device_id              TEXT NOT NULL REFERENCES devices (device_id),
    company_id             TEXT NOT NULL REFERENCES companies (company_id),
    employee_id            TEXT REFERENCES employees (employee_id),
    collected_at           TEXT NOT NULL,          -- ISO 8601 UTC
    collected_date         TEXT NOT NULL,          -- YYYY-MM-DD, for daily trends
    agent_version          TEXT,
    os_platform            TEXT NOT NULL,
    os_name                TEXT NOT NULL,
    os_version             TEXT NOT NULL,          -- as reported, e.g. "14.5", "11 23H2"
    os_version_major       INTEGER,                -- NULL when the format is unknown
    os_version_minor       INTEGER,
    os_version_patch       INTEGER,
    os_release             TEXT,                   -- Windows feature release, e.g. "23H2"
    os_build               TEXT,
    architecture           TEXT,
    hostname               TEXT,
    memory_total_bytes     INTEGER NOT NULL,
    memory_used_bytes      INTEGER NOT NULL,
    memory_free_bytes      INTEGER NOT NULL,
    memory_used_pct        REAL NOT NULL,
    battery_reported       INTEGER NOT NULL CHECK (battery_reported IN (0, 1)),
    battery_percentage     INTEGER,
    battery_charging_status TEXT,
    battery_condition      TEXT,
    battery_cycle_count    INTEGER,
    battery_full_charge_capacity INTEGER,
    UNIQUE (device_id, collected_at)
);

CREATE TABLE disk_volumes (
    snapshot_id       TEXT NOT NULL REFERENCES snapshots (snapshot_id),
    volume_name       TEXT NOT NULL,
    mount_point       TEXT NOT NULL,
    file_system       TEXT,
    size_bytes        INTEGER NOT NULL,
    available_bytes   INTEGER NOT NULL,
    free_pct          REAL NOT NULL,
    encrypted         INTEGER CHECK (encrypted IN (0, 1)),
    is_system_volume  INTEGER NOT NULL CHECK (is_system_volume IN (0, 1)),
    PRIMARY KEY (snapshot_id, mount_point)
);

CREATE TABLE compliance_results (
    snapshot_id  TEXT NOT NULL REFERENCES snapshots (snapshot_id),
    check_id     TEXT NOT NULL,
    severity     TEXT NOT NULL CHECK (severity IN ('low', 'medium', 'high', 'critical')),
    status       TEXT NOT NULL CHECK (status IN ('pass', 'fail')),
    PRIMARY KEY (snapshot_id, check_id)
);

CREATE TABLE installed_software (
    snapshot_id  TEXT NOT NULL REFERENCES snapshots (snapshot_id),
    name         TEXT NOT NULL,
    version      TEXT,
    publisher    TEXT,
    PRIMARY KEY (snapshot_id, name)
);

-- Key/value facts about the loaded dataset (as_of date, source hash, run id).
CREATE TABLE dataset_metadata (
    key    TEXT PRIMARY KEY,
    value  TEXT NOT NULL
);

CREATE INDEX idx_devices_company          ON devices (company_id);
CREATE INDEX idx_snapshots_company_time   ON snapshots (company_id, collected_at);
CREATE INDEX idx_snapshots_device_time    ON snapshots (device_id, collected_at);
CREATE INDEX idx_compliance_check_status  ON compliance_results (check_id, status);
CREATE INDEX idx_software_name            ON installed_software (name);
