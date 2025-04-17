NAME := avstack
INSTALL_STAMP := .install.stamp
UV := $(shell command -v uv 2> /dev/null)
PYFOLDERS := avstack tests third_party/mmdetection/CUSTOM third_party/mmdetection3d/CUSTOM
.DEFAULT_GOAL := help

.PHONY: help
help:
		@echo "Please use 'make <target>' where <target> is one of"
		@echo ""
		@echo "  install     install packages and prepare environment"
		@echo "  clean       remove all temporary files"
		@echo "  lint        run the code linters"
		@echo "  format      reformat code"
		@echo "  test        run all the tests"
		@echo ""
		@echo "Check the Makefile to know exactly what each target is doing."

install: $(INSTALL_STAMP)
$(INSTALL_STAMP): pyproject.toml uv.lock
		@if [ -z $(UV) ]; then echo "uv could not be found. See https://docs.astral.sh/uv/"; exit 2; fi
		# $(UV) sync --all-extras
		touch $(INSTALL_STAMP)

.PHONY: clean
clean:
		find . -type d -name "__pycache__" | xargs rm -rf {};
		rm -rf $(INSTALL_STAMP) .coverage .mypy_cache

.PHONY: lint
lint: $(INSTALL_STAMP)
		$(UV) run isort --profile=black --lines-after-imports=2 --check-only $(PYFOLDERS)
		$(UV) run black --check $(PYFOLDERS)  --diff
		$(UV) run flake8 --ignore=W503,E501 $(PYFOLDERS) 
		$(UV) run mypy $(PYFOLDERS)  --ignore-missing-imports
		$(UV) run bandit -r $(NAME) -s B608

.PHONY: format
format: $(INSTALL_STAMP)
		$(UV) run autoflake --remove-all-unused-imports -i -r $(PYFOLDERS) --exclude=__init__.py 
		$(UV) run isort --profile=black --lines-after-imports=2 $(PYFOLDERS) 
		$(UV) run black $(PYFOLDERS) 

.PHONY: test
test: $(INSTALL_STAMP)
		$(UV) run pytest ./tests/ --cov-report term-missing --cov-fail-under 0 --cov $(NAME)
