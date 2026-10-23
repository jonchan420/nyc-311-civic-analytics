# Complete months and data freshness

## Only complete months are used
Numbers for the current month aren't shown until the month is over and the next refresh has run. The tables are refreshed from NYC Open Data on the 5th of each month, and only months that had fully finished are kept. metadata.json records the last complete month as complete_through. The assistant refuses questions about months after it.

## Why partial months are excluded
A month still in progress has fewer requests than a full month, so it looks like a drop that isn't real. In this project, a forecast once used July 2026 when only about 70% of the month's requests had been recorded (226,542 records), which pulled every later prediction down. Since then every tool checks completeness first.

## What "last month" means
"Last month" means the most recent complete month in the data, which can differ from the calendar's last month right after a month ends. When they differ, the answer says which month it used.

## Gaps
If a borough has no rows at all for a month that should be complete, that is treated as missing data, not as zero complaints, and no answer is given.
