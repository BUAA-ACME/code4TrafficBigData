SELECT DISTINCT companies."Country", countries."GDPPC"
FROM countries JOIN companies
ON countries."Country" = companies."Country";

SELECT DISTINCT companies."Country", "Population"  
FROM ceos, countries, companies
WHERE ceos."Company" = companies."Company" 
AND countries."Country" = companies."Country";

SELECT "Name", "Age", "OneYrPay", "Shares", companies."Company", "Sales"
FROM ceos JOIN companies
ON ceos."Company" = companies."Company" 
WHERE companies."Country" = 'China' AND ( "Age">60 OR "Age" <50)
OR "OneYrPay" BETWEEN 20 AND 60
OR "Sales" > 1000000000
OR "Shares" >= 100;

SELECT "Country", COUNT( *) AS "CountOfCompanies"
FROM companies
GROUP BY "Country"
ORDER BY "CountOfCompanies" DESC;

SELECT "Country", COUNT( *) AS "CountOfCompanies"
FROM companies
GROUP BY "Country"
HAVING COUNT(*) > 20
ORDER BY "CountOfCompanies" DESC;

SELECT "Country", COUNT( *) AS "CountOfCompanies"
FROM companies
WHERE "Country" IN ( SELECT "Country" FROM countries WHERE "Population" > 80000000)
GROUP BY "Country";

SELECT "Country"
FROM countries
WHERE "Population" > 80000000
AND "Country" NOT IN ( SELECT "Country" FROM companies);

(SELECT 'Countries w/ Companies' AS "CountryGroup", 
        ROUND(AVG("GDPPC")::numeric,2) AS "AvgGDPPC"
   FROM Countries
  WHERE "Country" IN (SELECT "Country"
     FROM Companies))
  UNION
(SELECT 'Countries w/o Companies' AS "CountryGroup",
        ROUND(AVG("GDPPC")::numeric,2) AS "AvgGDPPC"
   FROM Countries
  WHERE "Country" NOT IN (SELECT "Country"
         FROM Companies));

SELECT "Country", MAX("Profits") AS "MaxProfits"
  FROM Companies
 GROUP BY "Country";

SELECT c."Country", "Company", "Profits"
  FROM Companies AS c JOIN (SELECT "Country", 
                                   MAX("Profits") AS "MaxProfits"
                              FROM Companies
                             GROUP BY "Country") AS a
    ON c."Country" = a."Country" 
       AND c."Profits" = a."MaxProfits";

SELECT c."Country", "Company",  
concat('$', ("Profits"::real/1000000000)::"numeric"(10,1), ' Billion') AS "Profits"
FROM Companies AS c JOIN (SELECT "Country", 
                                   MAX("Profits") AS "MaxProfits"
                              FROM Companies
                             GROUP BY "Country") AS a
    ON c."Country" = a."Country" 
       AND c."Profits" = a."MaxProfits"
 ORDER BY c."Profits" DESC;

CREATE TABLE ceos_age AS
SELECT *, concat(FLOOR("Age"/10)*10, '-',FLOOR("Age"/10+1)*10-1) AS "AgeGroup"
  FROM CEOs;

SELECT "AgeGroup", COUNT(*) AS "CEOCount"
  FROM ceos_age
 GROUP BY "AgeGroup"
 ORDER BY "AgeGroup";

SELECT "AgeGroup", ROUND(AVG("OneYrPay")::"numeric",2) AS "AvgPay"
  FROM ceos_age
 GROUP BY "AgeGroup"
 ORDER BY "AgeGroup";
