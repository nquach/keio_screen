Notes:
- functions.py contain the analysis functions for converting images and segmentation masks into lysis ratios
- keio_map.xlsx contains the Keio library map
- keio_names.py contains functions to convert a location into a strain name
- analyze_ecoli.py is the script that actually does the analysis for one data set
- analyze_slip.py is the script that does the analysis on one run of SLIP data
- prototype_functions.py is the script that contains prototyped functions. Not for use.
- SLIP_functions.py is a script containing functions to pass all SLIP images into neural nets (Must be used in the context of other CNN script!!)
- segment_SLIP.py is the script that segments all SLIP images for one Keio plate (both control and infection).
- interpolater.py is a script that converts plate reader OD600 readings to cfu/mL.