# How the forecasts work and their limits

## What is forecast
Two models forecast the citywide number of 311 requests per month. Gradient boosting is a machine-learning model that uses recent months, rolling averages, and the time of year. Seasonal naive simply predicts the same month's count from the year before. Forecasts are citywide only and come with no prediction intervals.

## Frozen forecasts
Each forecast is saved with a timestamp before its months happen and is never edited afterward. That prevents hindsight: a forecast can't be quietly improved after the real numbers are known. A GitHub Action scores the forecasts and freezes new ones on the 5th of each month.

## How accuracy is scored
Each month is scored with the most recent forecast made before that month's data could be complete. WAPE is the total absolute error divided by the total actual count; it works like an average percent miss. Forecast files with known bugs are listed in data/forecasts/superseded.csv and left out of the score.

## Validation and test results
In the 2025 validation year, the seasonal naive baseline beat both trained models: 2025 repeated its seasonal pattern closely. In 2026, complaints rose above 2025 levels and the trained models did better, because the baseline can only repeat last year.

## Known limits
The gradient boosting model can't predict above the range it was trained on, so in a year busier than any before it, it tends to come in low. Monthly forecasts also can't anticipate one-off events such as storms, parades, or major sports events, which would need daily data and event calendars.
