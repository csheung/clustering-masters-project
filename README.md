# Master's Project (Fall 2023)
# Uncovering Urban-Rural Disparities in Out-of-Hospital Cardiac Arrest (OHCA) Treatment Strategies through Unsupervised Machine Learning and Causal Analysis on the NEMSIS Dataset

This study looks to utilize clustering in understanding OHCA dataset by identifying potential clusters. It is hoped to gain better understanding toward the OHCA treatment and the urban-rural disparities, and stimulate discussions into causal relationships and possible interventions to improve rural prehospital cardiac arrest care.

This report conducted a retrospective analysis of a nationwide sample of OHCA incidents. Data were extracted from the 2020 National EMS Information System (NEMSIS) Version 3.4.0 Public-Release Research Dataset, administered by the MaineHealth organization and maintained by the Roux Institute at Northeastern University. This dataset is a convenience sample of data from prehospital patient care reports filed by EMS agencies in the United States (US). 

The 2020 dataset includes EMS incidents that occurred January 1 to December 31, 2020, submitted from 12,319 EMS agencies located in 50 states and territories. Each record in this dataset corresponds to one EMS unit dispatched to a reported incident. This dataset is de-identified and publicly accessible.

## Part 1. Data Pre-processing
The first step is to extract necessary feature data from respective dataframes such as epinephrine used, medications performed, medical protocols or procedures applied. 

a. Extract keys of events with epinephrine used
- Open the file of "1_medical_df.ipynb"

b. Run the code blocks through the Jupyter Notebook
- Read the csv file "FACTPCRMEDICATION_CA.csv" and turn it into a DataFrame
<img width="900" alt="Screenshot 2023-12-20 at 5 28 45 PM" src="https://github.com/csheung/clustering-masters-project/assets/99443055/1ecf2078-c4da-45b9-98f7-fc772c6c51d5">

- Drop unnecessary data based on the "eMedications_03" column
- Extract the event keys with the medical code in [317361, 328316] where 317361 equals each 0.1 MG/ML used and 328316 equals each 1 MG/ML used according to the RxNORM code.
<img width="711" alt="Screenshot 2023-12-21 at 7 25 14 PM" src="https://github.com/csheung/clustering-masters-project/assets/99443055/d9a881ca-0f1a-44fa-a922-212d34985601">

- Calculate the Frequency (column "EpinephrineFrequency") and the Total Amount (column "EpinephrineTotalAmount") of Epinephrine Used based on the dosage identified with the medical code.
- Keep the DataFrame with the necesaary columns of "EpinephrineUsed", "EpinephrineFrequency", "EpinephrineTotalAmount".
- Convert it in to the csv of the path 'clustering_data/epinephrine_cols.csv'.


