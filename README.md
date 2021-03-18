# Disaster Response Pipeline Project

### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database:
        make data/DisasterResponse.db
    - To run ML pipeline that trains classifier and saves:
        make models/classifier.pkl

2. Run the following command in the app's directory to run the web app.
    make web_app

3. Go to http://0.0.0.0:3001/

### Data Cleaning
Figure 8's data was stored in two CSV files: disaster_messages.csv
and  disaster_categories.csv. The data in each CSV had to be cleaned
and combined to form one DataFrame. However, the categories for a 
given message in disaster_categories.csv were given in the form of
semicolon separated lists; a message was in a category if the list
contained "{category}-1" and not in a category if it contained
"{category}-0". The method pd.Series.str.get_dummies method was very
useful for extracting category columns from the categories column.
However, there were redundant columns, since a category column's name 
was either {category}-0 or category-{1}; category-{0} and category-{1}
were always negations of each other. Redundant categories could be filtered
using pd.DataFrame.filter and its regex parameter. It was decided that
categories with a name like {category}-0 were kept, since child_alone-1
was never used as a category; this would result in 35 categories instead
of 36. Since columns in the form of {category}-0 were kept, they had to be
negated , which can be done by taking the exclusive-or operator (^) of the
columns and 1. For the final step of the data cleaning process, columns were
renamed by removing the "-0" at the end of each column and then merging
the data from messages and the modified categories DataFrames.

### How Imbalance Affects Training the Model
Categories with many examples usually tend to have high precision and high
recall, since any decent machine-learning model will just predict positive
most of the time. As a result, there will tend to be many true positives,
only a few false positives, and few false negatives, which is why the 
precision and recall will be high. Categories with only a few
examples, on the other hand, will have low precision and low recall, since
they have few true positives, false positives, and few true negatives.
