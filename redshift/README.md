# Amazon Redshift: POC & Presentation Notes

## Table of Contents

- [Schema & Data Generation](#schema--data-generation)
- [Analytical Queries](#analytical-queries)
- [Table Tuning: DISTKEY & SORTKEY](#table-tuning-distkey--sortkey)
- [Redshift vs. Snowflake: Storage & Optimization](#redshift-vs-snowflake-storage--optimization)
- [Redshift Scenarios & Differentiators](#redshift-scenarios--differentiators)
- [Pros & Cons](#pros--cons)
- [References](#references)

---

## Schema & Data Generation

### Create the Main Table

```sql
CREATE TABLE dev.public.op_systems (
    name        VARCHAR(200),
    country     VARCHAR(100),
    state       VARCHAR(100),
    age         INT,
    os          VARCHAR(20),
    is_rich     VARCHAR(50),
    is_insane   VARCHAR(50),
    is_nice     VARCHAR(50),
    reason      VARCHAR(500)
);
```

### Generate 5M Rows

```sql
-- Helper table for cross join
CREATE TABLE dev.public.multiplier_100 AS
WITH RECURSIVE nums(n) AS (
  SELECT 1
  UNION ALL
  SELECT n + 1 FROM nums WHERE n < 100
)
SELECT n FROM nums;

-- Create large table by cross joining
CREATE TABLE dev.public.op_systems_5m AS
SELECT a.*
FROM dev.public.op_systems a
CROSS JOIN dev.public.multiplier_100 m;

SELECT COUNT(*) FROM dev.public.op_systems_5m;
```

---

## Analytical Queries

### Query: Reasons Why People Use the OS

```sql
BEGIN;
DROP TABLE IF EXISTS tmp_sample;
CREATE TEMP TABLE tmp_sample AS
SELECT
  name,
  country,
  state,
  age,
  os,
  is_rich,
  is_nice,
  reason,
  md5(reason) AS reason_hash,
  SUBSTRING(reason, 1, 120) AS reason_snip
FROM dev.public.op_systems_5m
WHERE age BETWEEN 20 AND 60
  AND random() < 0.5;

ANALYZE tmp_sample;

WITH s AS (
  SELECT os, name
  FROM tmp_sample
  WHERE random() < 1.0
)
SELECT
  sub.os,
  sub.country,
  sub.reason_sample,
  sub.rows_in_bucket,
  sub.distinct_states,
  sub.distinct_names_joined,
  sub.avg_age,
  sub.avg_reason_len,
  AVG(sub.avg_age) OVER (PARTITION BY sub.os) AS avg_age_by_os
FROM (
  SELECT
    t.os,
    t.country,
    t.reason_snip AS reason_sample,
    COUNT(*) AS rows_in_bucket,
    COUNT(DISTINCT t.state) AS distinct_states,
    COUNT(DISTINCT s.name) AS distinct_names_joined,
    AVG(t.age) AS avg_age,
    AVG(LENGTH(t.reason)) AS avg_reason_len
  FROM tmp_sample t
  JOIN s USING (os)
  GROUP BY t.os, t.country, t.reason_snip
) sub
ORDER BY sub.rows_in_bucket DESC
LIMIT 200;
END;
```

### Query: Top Reasons

```sql
SELECT 
    reason::varchar AS reason,
    COUNT(*) AS occurrences
FROM dev.public.op_systems
GROUP BY reason::varchar
ORDER BY occurrences DESC
LIMIT 50;
```

---

## Table Tuning: DISTKEY & SORTKEY

### Create Tuned Table

```sql
DROP TABLE IF EXISTS dev.public.op_systems_5m_tuned;

CREATE TABLE dev.public.op_systems_5m_tuned
DISTSTYLE KEY
DISTKEY(os)
SORTKEY(os, country)
AS
SELECT
  name,
  country,
  state,
  age,
  os,
  is_rich,
  is_insane,
  is_nice,
  reason,
  md5(reason) AS reason_hash,
  SUBSTRING(reason, 1, 120) AS reason_snip
FROM dev.public.op_systems_5m;

ANALYZE dev.public.op_systems_5m_tuned;
```

### Query Against Tuned Table

```sql
BEGIN;

DROP TABLE IF EXISTS tmp_sample_tuned;

SELECT COUNT(*) AS base_total_rows 
FROM dev.public.op_systems_5m_tuned;

CREATE TEMP TABLE tmp_sample_tuned AS
SELECT
  name,
  country,
  state,
  age,
  os,
  is_rich,
  is_nice,
  reason,
  reason_hash,
  reason_snip
FROM dev.public.op_systems_5m_tuned
WHERE age BETWEEN 20 AND 60
  AND random() < 0.5;

ANALYZE tmp_sample_tuned;

SELECT COUNT(*) AS tmp_sample_rows 
FROM tmp_sample_tuned;

WITH s AS (
  SELECT os, name
  FROM tmp_sample_tuned
  WHERE random() < 1.0
),
agg AS (
  SELECT
    t.os,
    t.country,
    t.reason_snip AS reason_sample,
    COUNT(*) AS rows_in_bucket,
    COUNT(DISTINCT t.state) AS distinct_states,
    COUNT(DISTINCT s.name) AS distinct_names_joined,
    AVG(t.age) AS avg_age,
    AVG(LENGTH(t.reason)) AS avg_reason_len
  FROM tmp_sample_tuned t
  JOIN s USING (os)
  GROUP BY t.os, t.country, t.reason_snip
)
SELECT
  os,
  country,
  reason_sample,
  rows_in_bucket,
  distinct_states,
  distinct_names_joined,
  avg_age,
  avg_reason_len,
  AVG(avg_age) OVER (PARTITION BY os) AS avg_age_by_os
FROM agg
ORDER BY rows_in_bucket DESC
LIMIT 200;

END;
```

---

## Redshift vs. Snowflake: Storage & Optimization

### Columnar Storage

Columnar storage is not exclusive to Redshift. Many analytical databases use columnar formats for performance:

- **Amazon Redshift** (columnar + MPP + compression)
- **Snowflake**
- **BigQuery**
- **ClickHouse**
- **Apache Parquet** (file format)
- **Apache ORC**
- **Vertica**
- **Azure Synapse Analytics**
- **DuckDB**
- **MariaDB ColumnStore**
- Even **PostgreSQL** (with extensions like `cstore_fdw` or via Parquet FDWs)

**Why columnar?**  
Analytical queries read columns, not rows.  
Columnar layout enables data skipping, better compression, faster scans, and lower I/O—ideal for OLAP workloads.

### What is Exclusive to Redshift?

- **DISTKEY / SORTKEY** concepts (manual tuning)
- **Automatic Table Optimization** (sort/vacuum improvements)
- **Concurrency Scaling**
- **Spectrum** (querying S3 via Redshift)
- **RA3 instances + AQUA acceleration**

Columnar storage itself is generic and widely adopted.

---

## Redshift Scenarios & Differentiators

**When to use Redshift:**

- Analytics workloads (dashboards, BI, reports)
- Multi-TB datasets (logs, events, clickstreams)
- Need SQL over S3 (Spectrum)
- Apps ingesting data into Redshift (Kinesis, Firehose, ETL, DMS)

**What makes Redshift different?**

- Columnar storage → reduces IO
- Compression encodings → auto-optimized
- MPP execution → parallel query processing
- Spectrum → hybrid lakehouse (query S3)
- RA3 managed storage with hot cache → performance/cost win
- Smart Caching, Self-Optimizing features

---

## Redshift vs. Snowflake: Table Design

| Feature                | Redshift (Manual)                                        | Snowflake (Automated)                  |
|------------------------|----------------------------------------------------------|----------------------------------------|
| Distribution Keys      | Manual (DISTKEY)                                         | Automatic (micro-partitions)           |
| Sort Keys              | Manual (SORTKEY)                                         | Automatic (metadata, optional cluster) |
| Data Organization      | User-defined, requires tuning                            | Managed, auto-optimized                |
| Maintenance            | Vacuuming, tuning, schema design required                | Minimal, fully managed                 |
| Partition Pruning      | Zone maps (sort key)                                     | Metadata on micro-partitions           |
| Clustering             | N/A (manual sort key)                                    | Optional clustering key                |

---

## Pros & Cons

### Pros

- Deep AWS integration
- Explicit control over distribution and sort keys
- Serverless options available
- Designed for analytics and large-scale workloads
- Spectrum: query S3 directly

### Cons

- UI/config changes can take time to propagate
- Requires schema and key design for optimal performance
- Joins depend heavily on distribution (can require tuning)
- Maintenance: vacuuming, tuning, etc.

---

## References

- [Redshift Spectrum](https://docs.aws.amazon.com/redshift/latest/dg/c-using-spectrum.html)
- [RA3 Architecture](https://aws.amazon.com/redshift/features/ra3/)
- [Snowflake Clustering](https://docs.snowflake.com/en/user-guide/tables-clustering-keys)




