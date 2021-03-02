import sys
import pandas as pd
from sqlalchemy import create_engine

from sklearn.model_selection import train_test_split

from nltk.tokenize import word_tokenizer
from nltk.stem import PorterStemmer


def load_data(database_filepath: str):
    engine = create_engine("sqlite:///" + database_filepath)
    df: pd.DataFrame = pd.read_sql_table("messages", "sqlite:///" + database_filepath)

    category_colnames = df.columns
    return df['message'], 


def tokenize(text: str):
    tokens = word_tokenize(re.sub(r"\W", " ", text.lower()).strip())

    lemmatizer = WordNetLemmatizer()
    return [ lemmatizer.lemmatize(token) for token in tokens ]


def build_model():
    pipeline = Pipeline([
        ("vec", TfidfVectorizer(tokenizer = tokenize)),
        ("clf", MultiOutputClassifier(estimator = RandomForestClassifier()))
    ])

    parameters = { 
            "vec__ngram_range": ((1, 1), (1, 2)),
            "vec__max_df": (0.5, 0.75, 1.0),
            "vec__max_features": (None, 5000, 10000),
            "vec__use_idf": (True, False),
            "clf__estimator__n_estimators": (5, 50, 100, 250, 700),
            "clf__estimator__max_features": ("sqrt", "log2"),
            "clf__estimator__max_depth": (5, 8, 10)}

    cv = GridSearchCV(pipeline, grid_param = parameters)


def evaluate_model(model, X_test, Y_test, category_names):
    y_pred = model.predict(X_test)

    print(classification_report(Y_test, y_pred))


def save_model(model, model_filepath):
    pass


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2)

        print('Building model...')
        model = build_model()

        print('Training model...')
        model.fit(X_train, Y_train)

        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()
