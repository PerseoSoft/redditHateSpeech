#!/bin/bash
# The --login ensures the bash configuration is loaded,
# enabling Conda.

# Enable strict mode.
set -euo pipefail
# ... Run whatever commands ...

# Temporarily disable strict mode and activate conda:
set +euo pipefail
source activate hateSpeech

# Re-enable strict mode:
set -euo pipefail

# Run Jupyterlab
cd /opt/notebooks/clustering

jupyter lab --ip=0.0.0.0 --no-browser --allow-root --NotebookApp.token='' --NotebookApp.password=''
