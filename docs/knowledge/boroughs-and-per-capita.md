# Boroughs, citywide totals, and per-capita rates

## Borough counts leave out requests with no borough
Each request is assigned to one of the five boroughs from where it was reported. A small share has no borough recorded: about 0.11% in a typical month, and at most 2.61% (April 2020). Borough counts leave those out, so the five borough totals add up to slightly less than the citywide total.

## Citywide answers
Citywide totals come from monthly_citywide_counts.csv and include requests with no borough. A citywide breakdown by complaint type adds up the five boroughs instead, so answers report both numbers when they differ.

## Per-capita rates
Per-100,000 rates divide by 2020 Census populations from data/borough_population.csv. Per-capita rates can reorder boroughs: Brooklyn files the most complaints because it has the most people, while the Bronx files the most per resident.

## Limits of per-capita rates
Populations are from 2020 and don't reflect later change. Complaints are counted where they were reported, which can include commuters, visitors, and people reporting outside the borough where they live.

## Neighborhoods
This assistant works at the borough level. Neighborhood or ZIP code questions aren't supported yet; the project's separate reporting-gap analysis covers housing complaints by neighborhood for 2022 to 2024 only.
