#!/bin/bash

# Run Jupyterlab
cd /opt/notebooks/clustering
#sudo runuser -l jupyter -c "/opt/conda/bin/jupyter lab --ip=0.0.0.0 --no-browser"
jupyter lab --ip=0.0.0.0 --no-browser --allow-root
