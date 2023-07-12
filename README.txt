
This work is licensed under a Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 International License.

Author of Code: Jonathan Bedford, Ruhr University Bochum, Germany.
Contact Jonathan Bedford (jonathan.bedford@rub.de) if you have any problems using this code.
Date this code was shared:  July 12th, 2023

Use this reference:
Bedford, J. and Bevis, M., 2018. Greedy automatic signal decomposition and its application to daily GPS time series. Journal of Geophysical Research: Solid Earth, 123(8), pp.6992-7003.


Make sure you have the following packages:

	- numpy  (I am using 1.23.4)
	- tensorflow2  (I am using 2.9.1)
	- scipy (1.3.3)


See the Example_running_gratsid.ipynb




Final remarks:

- I strongly suggest removing obvious outliers (despiking) from your time series before running gratsid.

- GrAtSiD can handle gaps in the data, but it cannot handle NaN values (this crashes the algorithm)


