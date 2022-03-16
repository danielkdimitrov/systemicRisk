# Systemic Risk, Instructions 


+ Put the data files in the folder `data`. Follow the data formats in the sample files. It is important to set the same names for the institutions in each file. Input data includes:  

	+ CDS rates (in bps) in `cds`
	+ Total liabilities of the institutions are in `debt` 
	+ The ratio of Deposits for banks or policy liabilities (for insurers) to Total Liabilities in `other`
	+ VSTOXX index in `vola`
	
+ Model parameters are set in the class file `setParams`

	+ In `universeFull` set the names of the institutions to be analyzed. If any additional columns exist in the datafiles, they will not be processed
	+ Set the evaluation date in `lastDatePiT` and the start date (typically 2 years before the evaluation date) in `firstDatePiT`
	
+ To get results, run .py file `mainSystemicRisk`
+ Output of the file
	+ Default Probabilities and Default correlations
		+ Stored in the output class `pds` as dataframes `jpd` and `cpd`
	+ Marginal Expected Shortfall: 
		+ All output data is stored as dataframes in the resulting class `covarr`
		+ Two excel files are saved in the home folder: (1) tableMain.xlsx; (2)  NetworkESMatrix.xlsx containing the systemic risk estimates
+ Plots are automatically created for (1) Exposures, (2) Implied asset correlations, (3) Network MESs, (4) Conditional default probabilities 

+ Systemic risk rankings are based on a ranking by PC to ES (`tableMain['PCtoES99']`). This is similar to what we have in the Policy Note, you just need to execute in the end `tableMain['PCtoES99']*tableMain['ES99']['Sys']/100`  
