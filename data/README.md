Place your real training CSV here as:

`/Users/adarshvashistha/Desktop/genstone/data/crop_yield.csv`

Expected columns:

- `Area`
- `Item`
- `average_rain_fall_mm_per_year`
- `avg_temp`
- `pesticides_tonnes`
- `hg/ha_yield`

If the CSV is missing, `python3 src/train.py` will generate a demo dataset automatically so the app can be exercised end to end.
