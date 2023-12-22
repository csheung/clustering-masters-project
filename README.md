# Master's Project (Fall 2023)
## Uncovering Urban-Rural Disparities in Out-of-Hospital Cardiac Arrest (OHCA) Treatment Strategies through Unsupervised Machine Learning and Causal Analysis on the NEMSIS Dataset

This study looks to utilize clustering in understanding OHCA dataset by identifying potential clusters. It is hoped to gain better understanding toward the OHCA treatment and the urban-rural disparities, and stimulate discussions into causal relationships and possible interventions to improve rural prehospital cardiac arrest care.

This report conducted a retrospective analysis of a nationwide sample of OHCA incidents. Data were extracted from the 2020 National EMS Information System (NEMSIS) Version 3.4.0 Public-Release Research Dataset, administered by the MaineHealth organization and maintained by the Roux Institute at Northeastern University. This dataset is a convenience sample of data from prehospital patient care reports filed by EMS agencies in the United States (US). 

The 2020 dataset includes EMS incidents that occurred January 1 to December 31, 2020, submitted from 12,319 EMS agencies located in 50 states and territories. Each record in this dataset corresponds to one EMS unit dispatched to a reported incident. This dataset is de-identified and publicly accessible.


### Part 1. Data Pre-processing

The first step is to extract necessary feature data from respective dataframes such as epinephrine used, medications performed, medical protocols or procedures applied. 

a. Extract keys of events with epinephrine used
- Open the file of "1_medical_df.ipynb"

b. Run the code blocks through the Jupyter Notebook
- Read the csv file "FACTPCRMEDICATION_CA.csv" and turn it into a DataFrame
<img width="900" alt="Screenshot 2023-12-20 at 5 28 45 PM" src="https://github.com/csheung/clustering-masters-project/assets/99443055/1ecf2078-c4da-45b9-98f7-fc772c6c51d5">

- Drop unnecessary data based on the "eMedications_03" column
- Extract the event keys with the medical code in [317361, 328316] where 317361 equals each 0.1 MG/ML used and 328316 equals each 1 MG/ML used according to the RxNORM code.
- <img width="711" alt="Screenshot 2023-12-21 at 7 25 14 PM" src="https://github.com/csheung/clustering-masters-project/assets/99443055/d9a881ca-0f1a-44fa-a922-212d34985601">

- Calculate the Frequency (column "EpinephrineFrequency") and the Total Amount (column "EpinephrineTotalAmount") of Epinephrine Used based on the dosage identified with the medical code.
- Keep the DataFrame with the necesaary columns of "EpinephrineUsed", "EpinephrineFrequency", "EpinephrineTotalAmount".
- Convert it in to the csv of the path 'clustering_data/epinephrine_cols.csv'.


c. Extract keys of events with medical procedures applied from "FACTPCRPROCEDURE_CA.csv" 
- Open the file of "1_procedure_df_one_hot_encode.ipynb".
- <img width="880" alt="Screenshot 2023-12-21 at 7 33 07 PM" src="https://github.com/csheung/clustering-masters-project/assets/99443055/71eec5c0-5d6c-4251-aaed-71ce666d2c6c">

- Keep only the column of primary event keys "PcrKey" and the column of "eProcedures_03" and save it into the variable "procedure_df" for operations.

- Select the medical code related to airway management with a self-arranged csv "eProcedures_dict_airway.csv" only keeping those related code.

- Match those code (key) with the name (value) in the csv and turn it into a dictionary for modifying the column name later.

- Turn every row of extracted medical procedures into one-hot columns. Note that there could be rows with repeated keys as multiple procedures could be used in one event (look at key "71122902" example below).
- <img width="895" alt="Screenshot 2023-12-22 at 12 01 58 PM" src="https://github.com/csheung/clustering-masters-project/assets/99443055/7c005315-7ae4-47ff-8e07-37a71805544a">

- Combine the rows of the same events using logical OR operation for each column (look at the change in key "71122902" example below).
- <img width="895" alt="Screenshot 2023-12-22 at 12 21 36 PM" src="https://github.com/csheung/clustering-masters-project/assets/99443055/6fdccae5-00fe-4c1c-8828-de65a315bb20">

- No imputation was performed here because the csv only records those keys with medications or procedures performed. Other event rows were marked as 0 in the one-hot column representing nothing performed.

- Turn the combined one-hot-encoded DataFrame into a csv for recording.


### Part 2. Analyzing subsets with/without the disposition data (eDisposition_21)

- This section aims to compare the events with and without the disposition data and show that those with disposition data could be the more comprehensive and no-early-stop-recording event records.

- Utilize the "2_extract_analyze_disposition_subsets.ipynb" file

- Get the "eTimes_" EMS-time-related data from the "Pub_PCRevents_CA.csv" file.

- Split the dataset into two subsets by dividing them based on their eDisposition_21 data which could also be found in the "Pub_PCRevents_CA.csv" file.
  
- Subset / Group 1: The events NO disposition data (records of 'Not Recorded' or 'Not Applicable' with codes that start with a '7')
- Subset / Group 2: The events that HAS disposition data (other records that does not start with a '7'
- <img width="690" alt="Screenshot 2023-12-22 at 12 34 58 PM" src="https://github.com/csheung/clustering-masters-project/assets/99443055/48f92918-964d-4b52-a7fb-963e2ad3e87f">

- Create respective DataFrames for subsets combining with the preprocessed background data.
- Plot the precentages for missing values of the background data for comparison (Subset 1 was found to have more missing values for all columns including the outcome indicators than Subset 2).
- Plot the percentages for events with epinephrine used, death rates, and missing initial data for time points between two subsets.
- Visualize the subset differences using bar charts (like below).
- <img width="905" alt="Screenshot 2023-12-22 at 12 44 03 PM" src="https://github.com/csheung/clustering-masters-project/assets/99443055/9a7fff53-1dc7-486e-bff4-3ce21987d521">

- Save the primary keys for events that HAS disposition data for subsequent use.


### Part 3. Merging the selected columns related to EMS conditions to the main background DataFrame

- Utilize the "3_impute_bg_df_plus_combine_condition_cols.ipynb" file.

- Read ther EMS-condition-related columns from respective csv files.
<br>

| Column Added | CSV File |
|--------------|----------|
| ResuscitationAttempted | "FACTPCRARRESTRESUSCITATION_CA.csv" |
| PriorUseOfAED | "eArrest_07_AEDUse_11_ArrestRhythm_16_CPRReason.csv" |
| ArrestRhythm | "eArrest_07_AEDUse_11_ArrestRhythm_16_CPRReason.csv" |
| ReasonToStopCPR | "eArrest_07_AEDUse_11_ArrestRhythm_16_CPRReason.csv" |
| ProtocolsUsed | "FACTPCRPROTOCOL_CA.csv" |
| ProtocolAgeCategory | "FACTPCRPROTOCOL_CA.csv" |
| ArrestWitnessedBy | "FACTPCRARRESTWITNESS_CA.csv" |
| TypeOfCPRProvided | "FACTPCRARRESTCPRPROVIDED_CA.csv" |
| CardiacRhythmOnArrivalAtDestination | "FACTPCRARRESTRHYTHMDESTINATION_CA.csv" |


- Merge them to the background DataFrame based on the primary event key - "PcrKey" column.


### Part 4. One-hot-encoding the necessary columns for clustering

- Utilize the "4_add_onehot_encode_treatment.ipynb" file.

- Turn each categorical column into one-hot-encoded columns based on the categories it contains.

- Divide the column list of the processed DataFrame into two groups, one was the columns of continuous data, another one was the others of categorical data.

- Apply sklearn.impute.SimpleImputer to impute continuous columns with mean strategy and categorical columns with mode ("most_frequent") strategy.

- Save the imputed DataFrame into a csv file.

- Then, go to the "clustering_implementation" folder.


### Part 5. Preparing clustering-ready csv file

- Utilize the "5_combine_all_features_and_normalize_num_cols.ipynb" file.

- Merge the separately-handled epinephrine and medical procedure columns into the main DataFrame.

- Save the non-standardized version of DataFrame for subsequent use of clustering analysis.

- Standardize the data in continuous-typed columns.

- Drop unwanted columns e.g., columns starting with "USCensusRegion_".

- Get the primary event keys that HAS disposition data (from "keys_has_disposition_data.csv") and filter the large DataFrame.

- Save this one-hot-encoded and standardized DataFrame for clustering.


### Part 6. Clustering by running Python Scripts

- Navigate to the "umap_kmeans_implementation" folder.

- Utilize the "umap_km_array_job.bash" file to control job and task number to run respective Python Scripts on NEU Discovery Platform.

- Or further navigate to the "umap_km4" folder and run the "umap_km4" script of the same name.

- Run the program to generate UMAP then K-Means clustering strategy, and it would provide you with new DataFrame merged with the cluster label column and multiple clustering result visuals.

- The "umap_km4" was the one chosen for demonstration, code submission and report writing.

- Edit the scripts accordingly to test out variations of the program such as modifying the predetermined number of clusters.


### Part 7. Reference

[1] Tsao, C. W., Aday, A. W., Almarzooq, Z. I., Anderson, C. A. M., Arora, P., Avery, C. L., Baker-Smith, C. M., Beaton, A. Z., Boehme, A. K., Buxton, A. E., Commodore-Mensah, Y., Elkind, M. S. V., Evenson, K. R., Eze-Nliam, C., Fugar, S., Generoso, G., Heard, D. G., Hiremath, S., Ho, J. E., Kalani, R., … American Heart Association Council on Epidemiology and Prevention Statistics Committee and Stroke Statistics Subcommittee (2023). Heart Disease and Stroke Statistics-2023 Update: A Report From the American Heart Association. Circulation, 147(8), e93–e621. https://doi.org/10.1161/CIR.0000000000001123 

[2] Peters, G. A., Ordoobadi, A. J., Panchal, A. R., & Cash, R. E. (2023). Differences in Out-of-Hospital Cardiac Arrest Management and Outcomes across Urban, Suburban, and Rural Settings. Prehospital emergency care, 27(2), 162–169. https://doi.org/10.1080/10903127.2021.2018076 

[3] Blewer, A. L., McGovern, S. K., Schmicker, R. H., May, S., Morrison, L. J., Aufderheide, T. P., ... & Abella, B. S. (2018). Gender Disparities Among Adult Recipients of Bystander Cardiopulmonary Resuscitation in the Public. Circulation: Cardiovascular Quality and Outcomes, 11(8), e004710.4.

[4] Jadhav, S., & Gaddam, S. (2021). Gender and location disparities in prehospital bystander AED usage. Resuscitation, 158, 139–142. https://doi.org/10.1016/j.resuscitation.2020.11.006

[5] Jaramillo AP, Yasir M, Iyer N, Hussein S, Sn VP. Sudden Cardiac Death: A Systematic Review. Cureus. 2023 Aug 2;15(8):e42859. doi: 10.7759/cureus.42859. PMID: 37664320; PMCID: PMC10473441. 

[6] National EMS Information System, “2020 NEMSIS Dataset Manual,” Version 3, 2020. [Online]. Available: https://nemsis.org/wp-content/uploads/2021/05/2020-NEMSIS-RDS-340-User-Manual_v3-FINAL.pdf 

[7] Hozumi Y, Wang R, Yin C, Wei GW. UMAP-assisted K-means clustering of large-scale SARS-CoV-2 mutation datasets. Comput Biol Med. 2021 Apr;131:104264. doi: 10.1016/j.compbiomed.2021.104264. Epub 2021 Feb 22. PMID: 33647832; PMCID: PMC7897976. 

[8] Grubic, A. D., Testori, C., Sterz, F., Herkner, H., Krizanac, D., Weiser, C., ... & Schriefl, C. (2020). Bystander-initiated cardiopulmonary resuscitation and automated external defibrillator use after out-of-hospital cardiac arrest: Uncovering disparities in care and survival across the urban–rural spectrum. Resuscitation Plus, 5, 100054.23.

[9] Callaway CW. Epinephrine for cardiac arrest. Curr Opin Cardiol. 2013 Jan;28(1):36-42. doi: 10.1097/HCO.0b013e32835b0979. PMID: 23196774.
