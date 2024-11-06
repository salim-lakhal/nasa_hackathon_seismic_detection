�Seismic Detection across the Solar System�.

Parsing through seismic data collected on the Moon and Mars and figure out how to detect moonquakes and marsquakes!
Training set containing the following:

1. A catalog of quakes identified in the data
2. Seismic data collected by the Apollo (one day segments) or InSight (one hour segments) missions in miniseed and CSV format.
3. Plots of the trace and spectrogram for each day a quake has been identified.

Algorithm to identify these signals (using the catalog as a benchmark) and then
apply your algorithm to the seismic data in the "test" folder (Apollo 12, 15, and 16 for the Moon and other InSight events for Mars).

Each trace included in the test folder has a quake in it, there are no empty traces.

This main folder also has a Jupyter notebook that will help you get started on the data.

You can download the datasets here : https://www.spaceappschallenge.org/nasa-space-apps-2024/challenges/seismic-detection-across-the-solar-system/?tab=resources

** IMPORTANT **
Please make sure that your output catalog has at least the following headers:
filename
time_abs(%Y-%m-%dT%H:%M:%S.%f) or time_rel(sec)
