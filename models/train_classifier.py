import sys
import pandas as pd
import numpy as np
from sqlalchemy import create_engine
import joblib
import re

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report

import nltk
from nltk.tokenize import word_tokenize
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords

# nltk.download("punkt")
# nltk.download("stopwords")
# nltk.download("wordnet")

def load_data(database_filepath: str):
    """
    Loads data from database file database_filepath
    Parameters:
        database_filepath: path of database file to load data from
    Returns:
        Tuple consisting of 
    """
    engine = create_engine("sqlite:///" + database_filepath)
    df: pd.DataFrame = pd.read_sql_table("messages", "sqlite:///" + database_filepath)

    X = df["message"]
    Y = df.drop(["message", "original", "genre"], axis = 1)
    category_colnames = Y.columns

    return X, Y, Y.columns


def tokenize(text: str):
    """
    Tokenizes a document
    Parameters
        text: document to tokenize
    """
    tokens = word_tokenize(re.sub(r"\W", " ", text.lower()).strip())

    lemmatizer = WordNetLemmatizer()
    stemmer = PorterStemmer()
    return [ stemmer.stem(lemmatizer.lemmatize(token)) for token in tokens if token not in stopwords.words("english") ]


def build_model():
    """
    Builds machine-learning model
    """
    pipeline = Pipeline([
        ("vec", TfidfVectorizer(tokenizer = tokenize)),
        ("clf", MultiOutputClassifier(estimator = RandomForestClassifier()))
    ])

    parameters = { 
            "vec__ngram_range": ((1, 1), (1, 2)),
            "vec__max_df": (0.5, 0.75, 1.0),
            "vec__max_features": (None, 5000, 7500),
            "vec__use_idf": (True, False),
            "clf__estimator__n_estimators": (5, 50, 100),
            "clf__estimator__max_features": ("sqrt", "log2"),
            "clf__estimator__max_depth": (5, 8)}

    cv = GridSearchCV(pipeline, param_grid = parameters, verbose = 10, cv = 3)

    return cv


def evaluate_model(model, X_test, Y_test, category_names):
    """
    Evaluates performance of machine-learning model
    Parameters:
        model: Machine-learning model to evaluate
        X_test: Input testing data that the machine-learning model 
                will be predicting on
        Y_test: Testing data that the machine-learning model will
                be evaluating its performance against
        category_names: Category names that the machine-learning model will
                        be predicting
    """
    y_pred = model.predict(X_test)

    print(classification_report(Y_test, y_pred, target_names = category_names, labels = np.unique(y_pred)))


def save_model(model, model_filepath):
    """
    Saves model to file model_filepath using joblib
    Parameters:
        model: Machine-learning model to save
        model_filepath: path of file to save machine-learning model to
    """
    joblib.dump(model, model_filepath)


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print(f"Loading data...\n    DATABASE: {database_filepath}")
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2)

        print("Building model...")
        model = build_model()

        print("Training model...")
        model.fit(X_train, Y_train)

        print("Evaluating model...")
        evaluate_model(model, X_test, Y_test, category_names)

        print(f"Saving model...\n    MODEL: {model_filepath}")
        save_model(model, model_filepath)

        print("Trained model saved!")

    else:
        print("Please provide the filepath of the disaster messages database "\
              "as the first argument and the filepath of the pickle file to "\
              "save the model to as the second argument. \n\nExample: python "\
              "train_classifier.py ../data/DisasterResponse.db classifier.pkl")


if __name__ == "__main__":
    main()
