-- Each month's citywide total next to the five-borough sum; the difference is complaints
-- with no borough recorded, which is why borough totals don't add up to the citywide total.
SELECT
  CAST(c.month AS DATE) AS month,
  c.complaint_count AS citywide,
  b.borough_total,
  c.complaint_count - b.borough_total AS no_borough
FROM {monthly_citywide_counts} AS c
JOIN (
  SELECT CAST(month AS DATE) AS month, SUM(complaint_count) AS borough_total
  FROM {monthly_borough_category}
  GROUP BY 1
) AS b
  ON b.month = CAST(c.month AS DATE)
ORDER BY 1
