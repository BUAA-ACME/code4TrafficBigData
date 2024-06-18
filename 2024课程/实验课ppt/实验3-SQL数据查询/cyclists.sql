SELECT COUNT(*)  AS Count,
       Gender,
       DATE_FORMAT(w.Date, '%W')  AS DayOfWeek
  FROM Cyclists AS c JOIN Locations AS l ON c.LocationID= l.LocationID JOIN Weather AS w ON l.Date= w.Date
 WHERE w.MinTemp< 12
 GROUP BY DayOfWeek,
         Gender
 ORDER BY DayOfWeek,
         Gender;

SELECT COUNT(*) AS TotCount, Road, Gender
  FROM Cyclists AS c JOIN Locations AS l
    ON c.LocationID = l.LocationID
 GROUP BY Road, Gender;

SELECT COUNT(*) AS HelmetCount, Road, Gender
  FROM Cyclists AS c JOIN Locations AS l
    ON c.LocationID = l.LocationID
 WHERE Helmet = 'Yes'
 GROUP BY Road, Gender;

SELECT l.Road, c.Gender,
       COUNT(*)*1.0/AVG(TotCount) AS 'Helmet User Fraction'
  FROM Cyclists AS c
       JOIN Locations AS l ON c.LocationID = l.LocationID
       JOIN (SELECT COUNT(*) AS TotCount, Road, Gender
               FROM Cyclists AS cc JOIN Locations AS ll
                 ON cc.LocationID = ll.LocationID
              GROUP BY Road, Gender) AS t
    ON l.Road = t.Road AND c.Gender = t.Gender 
 WHERE Helmet = 'Yes'
 GROUP BY l.Road, c.Gender
 ORDER BY l.Road, c.Gender;

CREATE VIEW Total AS
SELECT COUNT(*) AS TotCount, Road, Gender
  FROM Cyclists AS c JOIN Locations AS l
    ON c.LocationID = l.LocationID
 GROUP BY Road, Gender;

CREATE VIEW Helmet AS
SELECT COUNT(*) AS HelmetCount, Road, Gender
  FROM Cyclists AS c JOIN Locations AS l
    ON c.LocationID = l.LocationID
 WHERE Helmet = 'Yes'
 GROUP BY Road, Gender;

SELECT h.Road, h.Gender,
       HelmetCount*1.0/TotCount AS 'Helmet User Fraction'
  FROM Helmet AS h, Total AS t
 WHERE h.Road = t.Road AND h.Gender = t.Gender
 ORDER BY h.Road, h.Gender;

 SELECT PersonID,
       CASE Helmet
            WHEN 'Yes' THEN 1.0
            ELSE 0.0
       END AS Hel
  FROM Cyclists;

SELECT Road, Gender,
       AVG(CASE Helmet
             WHEN 'Yes' THEN 1.0
             ELSE 0.0
           END) AS 'Helmet User Fraction'
  FROM Cyclists AS c JOIN Locations AS l
    ON c.LocationID = l.LocationID
 GROUP BY Road, Gender
 ORDER BY Road, Gender;

ALTER TABLE Locations
  ADD CyclistCount INT;


UPDATE Locations AS l JOIN (SELECT LocationID, COUNT(*) AS count
               FROM Cyclists
              GROUP BY LocationID) AS c
ON l.LocationID = c.LocationID
SET l.CyclistCount = c.count;

-- DELETE FROM Cyclists
-- WHERE LocationID NOT IN (SELECT LocationID FROM Locations);

-- DELETE FROM locations
-- WHERE Date = (SELECT a.mindate FROM
-- (SELECT MIN(Date) as mindate
-- FROM locations) a);

