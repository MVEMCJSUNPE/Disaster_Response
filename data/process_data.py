import sys
import pandas as pd
from sqlalchemy import create_engine

def load_data(messages_filepath: str, categories_filepath: str) -> pd.DataFrame:
    """
    Loads data from csv into pandas dataframe

    Parameters:
        messages_filepath: CSV file to get messages
        categories_filepath: CSV file to get categories
    """
    messages = pd.read_csv(messages_filepath, index_col = "id")
    categories = pd.read_csv(categories_filepath, index_col = "id")

    categories_intermed = categories["categories"].str.get_dummies(";")
    categories_colnames = set( col[ : -2] for col in categories.columns )

    categories = category_intermed.filter(regex = r"[a-zA-Z_]+-1")

    # Loop to add in all categories that are 0
    for col in categories_colnames:
        if col not in categories.columns:
            categories[col + "-1"] = 0

    categories.rename(columns = lambda col: col[ : col.index("-1")], inplace = True)

    df = pd.merge(messages, categories, on = "id")

    return df


def clean_data(df: pd.DataFrame):
    return df.drop_duplicates()


def save_data(df: pd.DataFrame, database_filename: str):
    engine = create_engine(f"sqlite:///{database_filename}")
    df.to_sql("messages", engine, index = False)


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)

        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)

        print('Cleaned data saved to database!')

    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()
