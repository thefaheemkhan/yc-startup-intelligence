-- YC Startup Intelligence: relational schema (SQLite + PostgreSQL compatible types).
-- Tables are limited to what the source data supports. There is no founders table and no
-- status-history table because the source has neither (status is a single current snapshot).
DROP TABLE IF EXISTS company_tags;
DROP TABLE IF EXISTS company_locations;
DROP TABLE IF EXISTS companies;
DROP TABLE IF EXISTS locations;
DROP TABLE IF EXISTS industries;
DROP TABLE IF EXISTS batches;

CREATE TABLE batches (
    batch_id INTEGER PRIMARY KEY,
    batch TEXT NOT NULL UNIQUE,
    season TEXT,
    year INTEGER,
    code TEXT,
    batch_order INTEGER,
    approx_start_date TEXT,
    batch_status TEXT
);

CREATE TABLE industries (
    industry_id INTEGER PRIMARY KEY,
    industry TEXT NOT NULL,
    subindustry TEXT,
    UNIQUE (industry, subindustry)
);

CREATE TABLE locations (
    location_id INTEGER PRIMARY KEY,
    place_raw TEXT NOT NULL UNIQUE,
    city TEXT,
    region TEXT,
    country TEXT,
    is_remote INTEGER NOT NULL
);

CREATE TABLE companies (
    company_id INTEGER PRIMARY KEY,
    slug TEXT NOT NULL UNIQUE,
    name TEXT NOT NULL,
    website TEXT,
    yc_url TEXT,
    one_liner TEXT,
    long_description TEXT,
    batch_id INTEGER REFERENCES batches (batch_id),
    industry_id INTEGER REFERENCES industries (industry_id),
    status TEXT NOT NULL,
    is_operating INTEGER NOT NULL,
    stage TEXT,
    team_size INTEGER,
    team_size_outlier INTEGER NOT NULL,
    is_hiring INTEGER NOT NULL,
    is_nonprofit INTEGER NOT NULL,
    is_top_company INTEGER NOT NULL,
    ai_tag INTEGER NOT NULL,
    ai_keyword INTEGER NOT NULL,
    is_ai INTEGER NOT NULL,
    is_remote INTEGER NOT NULL,
    launched_at TEXT,
    primary_location_id INTEGER REFERENCES locations (location_id)
);

CREATE TABLE company_locations (
    company_id INTEGER NOT NULL REFERENCES companies (company_id),
    location_id INTEGER NOT NULL REFERENCES locations (location_id),
    is_primary INTEGER NOT NULL,
    PRIMARY KEY (company_id, location_id)
);

CREATE TABLE company_tags (
    company_id INTEGER NOT NULL REFERENCES companies (company_id),
    tag TEXT NOT NULL,
    PRIMARY KEY (company_id, tag)
);

CREATE INDEX idx_companies_batch ON companies (batch_id);
CREATE INDEX idx_companies_industry ON companies (industry_id);
CREATE INDEX idx_companies_status ON companies (status);
CREATE INDEX idx_company_tags_tag ON company_tags (tag);
