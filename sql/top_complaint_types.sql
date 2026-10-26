-- The 15 complaint types with the most recorded complaints: the Trends tab's choices.
SELECT
  complaint_type,
  SUM(complaint_count) AS complaints
FROM {monthly_borough_category}
GROUP BY complaint_type
ORDER BY complaints DESC, complaint_type
LIMIT 15
