
#°¸Àý1
SELECT gid, name, geom
FROM highways WHERE name = 'US Route 1' AND state = 'MD';

WITH midtable AS (
SEKECT ST_Union(geom) as geom FORM highways
WHERE name = 'US Route 1' AND state = 'NJ'
)
SELECT ST_Buffer(geom::geography,1000) FROM midtable

WITH midtable AS (
SELECT ST_Union(geom) as geom FROM highways
WHERE name = 'US Route 1' AND state = 'NJ'
)
SELECT * FROM restaurants WHERE ST_Within(geom,(SELECT ST_Buffer(geom::geography,1000) FROM midtable)::geometry)

SELECT lu_franchises.franchise AS Fullname,finaltable.franchise AS Shortname,count FROM
(WITH midtable AS (
SELECT ST_Union(geom) as geom FROM highways
WHERE name = 'US Route 1' AND state = 'NJ'
)
SELECT franchise,COUNT(*) FROM restaurants WHERE ST_Within(geom,(SELECT ST_Buffer(geom::geography,1000) FROM midtable)::geometry)
GROUP BY franchise) AS finaltable
LEFT JOIN lu_franchises
ON finaltable.franchise = lu_franchises.id


#°¸Àý2
CREATE TABLE trajectory
(
gid CHARACTER VARYING,
tms INTEGER,
lon NUMERIC,
lat NUMERIC
);
COPY trajectory FROM 'C:/Users/54318/Desktop/Spatial Data/trajectory.csv' DELIMITER AS ',';

ALTER TABLE trajectory ADD COLUMN trace_geom geometry(Point,4326);
UPDATE trajectory SET trace_geom = ST_SetSRID(ST_MakePOINT(lon,lat),4326);

CREATE INDEX idx_trace ON trajectory USING gist(trace_geom);

SELECT * FROM trajectory p
CROSS JOIN LATERAL (
    SELECT r.id AS trace_id, r.osm_name AS raod_name, r.geom AS road_geom
	FROM anshan_road r
	ORDER BY r.geom <-> p.trace_geom
	LIMIT 1
)midtable;