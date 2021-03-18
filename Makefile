PYCMD = python

DATA = data/disaster_messages.csv data/disaster_categories.csv

models/classifier.pkl: models/train_classifier.py data/DisasterResponse.db
	$(PYCMD) $^ $@

data/DisasterResponse.db: data/process_data.py $(DATA)
	$(PYCMD) $^ $@

.PHONY: web_app clean clean_model clean_db nltk

web_app: app/run.py # models/classifier.pkl data/DisasterResponse.db
	$(PYCMD) app/run.py

nltk:
	$(PYCMD) -c "import nltk \
	    nltk.download('punkt', download_dir = 'model') \
	    nltk.download('stopwords', download_dir = 'model') \
	    nltk.download('wordnet', download_dir = 'model')"

clean: clean_model clean_db

clean_model:
	$(RM) models/classifier.pkl

clean_db:
	$(RM) data/DisasterResponse.db
