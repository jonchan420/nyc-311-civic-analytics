# Complaint categories and renamed labels

## Why labels are combined
The same kind of complaint can appear under different labels. Some differ only in spelling or capitalization, such as "ELEVATOR" and "Elevator". Others were renamed by the agency, so the old label stops and a new one takes over. Counting them separately would make a rename look like one category collapsing and another appearing from nothing.

## The category catalog
data/live/category_catalog.csv maps all 274 raw labels to 238 categories: 16 spelling variants and 20 renames. Each category is named by its most-used current spelling. Every label with at least 100 requests that stopped appearing between January 2020 and September 2026 was reviewed by checking the agency, the meaning, and the month-by-month handover.

## Examples of renames
In August 2026 the Department of Environmental Protection renamed "Sewer" to "Sewer Maintenance", "Water System" to "Water Maintenance", "Hazardous Materials" to "Hazardous Material", and "Plant" to "Waste Water Treatment Plant". In September and October 2021 the Department of Sanitation relabeled many complaints; for example "Dirty Conditions" became "Dirty Condition" and "Missed Collection (All Materials)" became "Missed Collection". "Smoking" became "Smoking or Vaping", with the new label dominant from April 2024.

## Labels with no single successor
Some labels ended without one clear replacement and are kept separate. "Sanitation Condition" ended in September 2021 and its mix of complaints has no single successor. "Vending" ended in January 2021, and "Vendor Enforcement" only starts in March 2023, so it is not treated as a rename.

## Flags on results
new_in_b means a category had no requests in the earlier period; it may be new or an unrecognized rename. absent_in_b means it had none in the later period. renamed means the row combines an old and a new label. unreviewed means a label appeared after the catalog was built, so a rename would not be recognized yet.
