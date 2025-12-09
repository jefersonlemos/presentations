begining



features

SORT KEY and DIST KEY

Materialized view 

checar se essas coisas são específicas do Redshift


queries

```
-- how many rows
SELECT COUNT(*) FROM dev.public.op_systems;

-- distribution by OS
SELECT os, COUNT(*) AS total
FROM dev.public.op_systems
GROUP BY os
ORDER BY total DESC;

-- how many insane vs nice per OS
SELECT os, is_insane, is_nice, COUNT(*) AS total
FROM dev.public.op_systems
GROUP BY os, is_insane, is_nice
ORDER BY os, total DESC;

-- avg age per OS and country
SELECT os, country, AVG(age) AS avg_age, COUNT(*) AS total
FROM dev.public.op_systems
GROUP BY os, country
HAVING COUNT(*) > 10
ORDER BY total DESC;

```
pros and cons

