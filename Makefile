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

.PHONY: ga
ga:
	python main.py --method nsga2 --map maps/grid_asymetric.txt --plot-path --plot-distances --save-plot nsga2

.PHONY: vns
vns:
	python main.py --method vns --map maps/grid_asymetric.txt --plot-path --plot-interpolation --plot-distances --save-plot vns

.PHONY: movns
movns:
	python main.py --map maps/grid_asymetric.txt --plot-path --plot-interpolation --plot-distances --save-plot movns1
	
.PHONY: movns-large
movns-large:
	python main.py --map maps/dispersed_large.txt --plot-path --plot-interpolation --plot-distances --save-plot movnslarge

.PHONY: clean-output
clean-output:
	rm - rf out/*