# # Ingest WFM data
# python ingest_data.py \
#   --input-dir /home/geuba03p/PyProjects/WINGS/data/fixed_pop_50\
#   --output /home/geuba03p/PyProjects/WINGS/data/combined_wfm.csv

# Ingest ABM data  
python ingest_data.py \
  --input-dir /home/geuba03p/PyProjects/WINGS/data/compare_spread_features \
  --output /home/geuba03p/PyProjects/WINGS/data/combined_abm.csv

# Generate figures (separate directories)
python plot_wings.py --model wfm --input /home/geuba03p/PyProjects/WINGS/data/combined_wfm.csv
python plot_wings.py --model abm --input /home/geuba03p/PyProjects/WINGS/data/combined_abm.csv