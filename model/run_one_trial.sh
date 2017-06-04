#!/bin/bash

python train_model_important_counties_deep2.py 2>&1 | tee ~/cs231n-satellite-images-models/runs/run7__deeper2__soybean/train_log
sudo poweroff
