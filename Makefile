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
	python main.py --map maps/dispersed_large.txt --num-agents 4 --budget 150 --num-iter 300 --speeds 1 1 1 1.5 --budget 150 150 150 200

.PHONY: clean-output
clean-output:
	rmdir /S /Q .\out
	mkdir .\out

.PHONY: clean-tests
clean-tests:
	rmdir /S /Q .\tests
	mkdir .\tests

.PHONY: clean-images
clean-images:
	rmdir /S /Q .\imgs\movns\paths
	mkdir .\imgs\movns\paths

.PHONY: clean
clean: clean-output clean-tests clean-images