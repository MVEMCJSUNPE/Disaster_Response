PYCMD = python

DATA = data/disaster_messages.csv data/disaster_categories.csv

models/classifier.pkl: models/train_classifier.py data/DisasterResponse.db
	$(PYCMD) $^ $@

data/DisasterResponse.db: data/process_data.py $(DATA)
	$(PYCMD) $^ $@

.PHONY: web_app clean clean_model clean_db

web_app: run.py
	$(PYCMD) run.py

clean: clean_model clean_db

clean_model:
	$(RM) models/classifier.pkl

clean_db:
	$(RM) data/DisasterResponse.db
