deps:
	pip-compile --output-file requirements.txt requirements.in
	pip-sync

tests:
	pytest

pro:
	python -m cProfile -o $(exp).cprofile match_dialogs.py $(exp)
