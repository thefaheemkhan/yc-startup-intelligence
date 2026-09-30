# Data Quality Report

- Raw rows: **6,260**; clean rows: **6,260**; columns: **33**
- Duplicates: {'company_id': 0, 'slug': 0, 'name_case_insensitive_extra_rows': 101}
- Batch years: [2005, 2027]; launched_at: ['2010-01-17', '2026-09-29']

## Cleaning decisions

| Step | Rows | Decision |
|---|---:|---|
| duplicate_ids | 0 | Dropped repeated company ids (kept last). |
| unknown_status | 0 | Status outside {Active, Inactive, Acquired, Public} recoded to 'Unknown'. |
| industry_unspecified | 18 | Kept as its own category 'Unspecified' (source value); exclude from industry rate metrics. |
| unparsed_batch | 1 | Batch not '<Season> <YYYY>' (e.g. 'Unspecified'): season/year/code null, row retained. |
| team_size_invalid | 134 | team_size <= 0 set to null (invalid headcount). |
| team_size_outlier | 13 | team_size > 3038 (3xIQR fence, log1p scale) flagged and retained; use robust statistics. |
| team_size_missing | 245 | Missing team_size left null (not imputed). |
| launched_at_out_of_range | 0 | launched_at outside [2005, as_of+1d] set to null. |
| no_physical_location | 197 | No non-remote location (missing or remote-only): primary_city/country left null, not imputed. |
| shared_names | 191 | Rows sharing a case-insensitive name kept: distinct company_id/slug. |

## Missing values (%, top 12)

| Column | Missing % |
|---|---:|
| subindustry | 21.65 |
| tags | 13.85 |
| long_description | 6.71 |
| primary_region | 5.77 |
| team_size | 3.91 |
| primary_city | 3.23 |
| primary_country | 3.15 |
| one_liner | 2.56 |
| all_locations | 2.44 |
| location_entries | 2.44 |
| website | 0.59 |
| batch_season | 0.02 |
