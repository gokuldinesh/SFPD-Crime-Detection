# SFPD-Crime-Detection

# Overview:

From 1934 to 1963, San Francisco was infamous for housing some of the world's most
notorious criminals on the inescapable island of Alcatraz.
Today, the city is known more for its tech scene than its criminal past. But, with rising
wealth inequality, housing shortages, there is no scarcity of crime in the city by the bay.
From Sunset to SOMA, and Marina to Excelsior, this dataset provides nearly 12 years
of crime reports from across all of San Francisco's neighborhoods. Given time and
location, you must predict the category of crime that occurred.

# Dataset:

This dataset contains incidents derived from SFPD Crime Incident Reporting system.
The data ranges from 1/1/2003 to 5/13/2015. The training set and test set rotate every
week, meaning week 1,3,5,7... belong to test set, week 2,4,6,8 belong to training set.

● Dates​ - timestamp of the crime incident
● Category​ - category of the crime incident . This is the target variable you are
going to predict.
● Descript​ - detailed description of the crime incident (only in Train.csv)
● DayOfWeek​ - the day of the week
● PdDistrict​ - name of the Police Department District
● Resolution​ - how the crime incident was resolved
● Address​ - the approximate street address of the crime incident
● X​ - Longitude
● Y​ - Latitude

# Output:

Each row contains
the ID of the incident, and then the probabilities of each respective crime occurring
given the incident’s details, as shown below:
Id ARSON …. WARRANTS WEAPON
LAWS
20 0.4 0.5 0.1
The order of the columns should be:
[Id,ARSON,ASSAULT,BAD CHECKS,BRIBERY,BURGLARY,DISORDERLY
CONDUCT,DRIVING UNDER THE
INFLUENCE,DRUG/NARCOTIC,DRUNKENNESS,EMBEZZLEMENT,EXTORTION,FA
MILY OFFENSES,FORGERY/COUNTERFEITING,FRAUD,GAMBLING,KIDNAPPING,
LARCENY/THEFT,LIQUOR LAWS,LOITERING,MISSING
PERSON,NON-CRIMINAL,OTHER OFFENSES,PORNOGRAPHY/OBSCENE
MAT,PROSTITUTION,RECOVERED VEHICLE,ROBBERY,RUNAWAY,SECONDARY
CODES,SEX OFFENSES FORCIBLE,SEX OFFENSES NON FORCIBLE,STOLEN
PROPERTY,SUICIDE,SUSPICIOUS OCC,TREA,TRESPASS,VANDALISM,VEHICLE
THEFT,WARRANTS,WEAPON LAWS]
These columns represent the classes in this multi-classification.

The metric used for evaluating the algorithm will be multi-class logarithmic loss.
Check out sklearn documentation for information on how to implement this metric.
