set -e

poetry install
python scripts/make-reqs.py custom.tool.conda.dependencies > requirements.conda
python scripts/make-reqs.py custom.tool.conda.cpu-dependencies > requirements-cpu.conda
conda install --file=requirements.conda
conda install --file=requirements-cpu.conda
