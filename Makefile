isort:
	@echo "\n${BLUE}Running automatic formatting import...${NC}\n"
	@isort --recursive --settings-path=setup.cfg .
.PHONY: isort

black:
	@echo "\n${BLUE}Running automatic formatting python...${NC}\n"
	@black .
.PHONY: black

linting:
	@echo "\n${BLUE}Running Flake8 against service files...${NC}\n"
	@flake8 .
	@echo "\n${BLUE}Running Bandit against source files...${NC}\n"
	@bandit -r --ini setup.cfg
.PHONY: linting

clean:
	@rm -rf .pytest_cache .coverage .mypy_cache coverage.xml reports runs .ipynb_checkpoints wandb ./*/.ipynb_checkpoints ./*/wandb
	@find . -type d -name __pycache__ -exec rm -r {} \+
