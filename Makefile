.ONESHELL:
# To begin the project, run the following command: make setup
# To activate the virtual environment, run the following command: make activate
# To run the project, run the following command: make run

VENV := .venv
REQUIREMENTS := requirements.txt

.PHONY: all
all: venv

.PHONY: venv
venv: # TODO: Needs fixing. Not activating the virtual environment.
	if not exist $(VENV) ( \
		python -m venv $(VENV) && \
		call .\$(VENV)\Scripts\activate && \
		python -m pip install --upgrade pip \
	) else ( \
		echo Virtual environment already exists. \
	)

.PHONY: activate
activate: # TODO: Needs fixing. Not activating the virtual environment.
	call .\$(VENV)\Scripts\activate

.PHONY: requirements
requirements:
	pip freeze > $(REQUIREMENTS)

.PHONY: install
install: requirements.txt
	pip install -r $(REQUIREMENTS)

.PHONY: run
run:
	python main.py --map maps/dispersed_large.txt --num-agents 3 --num-iter 1000 --speeds 2 1 1 --budget 300 300 300

.PHONY: remove_out_dir
remove_out_dir:
	if exist .\out ( \
		rmdir /S /Q .\out \
	)

.PHONY: create_out_dir
create_out_dir:
	if not exist .\out ( \
		mkdir .\out \
	)

.PHONY: clean_output
clean_output: remove_out_dir create_out_dir

.PHONY: remove_tests_dir
remove_tests_dir:
	if exist .\tests ( \
		rmdir /S /Q .\tests \
	)

.PHONY: create_tests_dir
create_tests_dir:
	if not exist .\tests ( \
		mkdir .\tests \
	)

.PHONY: clean_tests
clean_tests: remove_tests_dir create_tests_dir

.PHONY: remove_imgs_dir
remove_imgs_dir:
	if exist .\imgs ( \
		rmdir /S /Q .\imgs \
	)

.PHONY: create_imgs_dir
create_imgs_dir:
	if not exist .\imgs ( \
		mkdir .\imgs && \
		mkdir .\imgs\movns && \
		mkdir .\imgs\movns\paths && \
		mkdir .\imgs\movns\animations \
	)

.PHONY: clean_images
clean_images: remove_imgs_dir create_imgs_dir

.PHONY: create_dirs
create_dirs: create_out_dir create_tests_dir create_imgs_dir

.PHONY: remove_dirs
remove_dirs: remove_out_dir remove_tests_dir remove_imgs_dir

.PHONY: clean
clean: clean_output clean_tests clean_images

.PHONY: test
test: clean run

.PHONY: setup
setup: venv install