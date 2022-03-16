# Systemic Risk, Instructions 


+ Put the data files in the folder `Data`. Follow the data formats in the sample files. It is important to set the same names for the institutions in each file. Input data includes:  

	+ CDS rates (in bps) in `cds`
	+ Total liabilities of the institutions are in `debt` 
	+ The ratio of Deposits (Policy Liabilities for Insurers) to Total Liabilities in `other`
	+ VSTOXX index implied volatility in `vola`
	
+ Model parameters are set in the class file `setParams`

	+ In `universeFull` set the names of the institutions to be analyzed. If any additional columns exist in the datafiles, they will not be processed
	+ Set the evaluation date in `lastDatePiT` and the start date (typically 2 years before the evaluation date) in `firstDatePiT`
	
+ To get results, run .py file `mainSystemicRisk`
+ Output of the file
	+ Default Probabilities and Default correlations
		+ Stored in the output class `pds` as as dataframes `jpd` and `cpd`
	+ Marginal Expected Shortfall: 
		+ All output data is stored as dataframes in the resulting class `covarr`
		+ Marginal Expected Shortfall: two excel files are saved in the home folder: (1) tableMain.xlsx; (2)  NetworkESMatrix.xlsx