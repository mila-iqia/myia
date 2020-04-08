set -e

poetry install
python scripts/make-reqs.py custom.tool.conda.dependencies > requirements.conda
python scripts/make-reqs.py custom.tool.conda.gpu-dependencies > requirements-gpu.conda
conda install --file=requirements.conda
conda install --file=requirements-gpu.conda
