# What the data contains

## Source
The data comes from NYC Open Data: the 311 service requests dataset (erm2-nwe9), from January 2020 onward. A service request is a complaint or request someone filed with 311 by phone, online, or through the mobile app.

## The tables the assistant reads
The assistant reads small summary tables in data/live/, not individual records. monthly_borough_category.parquet has one row per month, borough, and complaint type, with a count. monthly_citywide_counts.csv has one total per month for the whole city. metadata.json records when the tables were refreshed and the last complete month.

## What one row means
A count is the number of service requests created in that month, in that borough, with that complaint type. If a complaint type has no row for a month, it had zero requests that month. Counts are by the month a request was created, not when it was closed.

## What is not in these tables
The tables have no addresses, no individual requests, no response or closing times, no reporting channel (phone, online, app), and no neighborhoods or ZIP codes. Questions that need any of those can't be answered from them.
