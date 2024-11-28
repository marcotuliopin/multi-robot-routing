VENV := .venv
REQUIREMENTS := requirements.txt

.PHONY: all
all: venv

.PHONY: venv
venv:
	if not exist $(VENV) ( \
		python -m venv $(VENV) && \
		.\$(VENV)\Scripts\activate && \
		python -m pip install --upgrade pip && \
		pip install -r $(REQUIREMENTS) \
	) else ( \
		echo Virtual environment already exists. \
	)

.PHONY: install
install: venv
	pip install -r $(REQUIREMENTS)

.PHONY: run-ga
run-ga:
	python main.py --method nsga2 --map maps/grid_asymetric.txt --plot-path --plot-distances