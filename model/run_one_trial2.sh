#!/bin/bash

#python train_model_important_counties_deep3.py 2>&1 | tee ~/cs231n-satellite-images-models/runs/run8__deeper3__soybean/important_counties/train_log
#python train_model_important_counties_deep4.py 2>&1 | tee ~/cs231n-satellite-images-models/runs/run9__deeper4__soybean/important_counties/train_log
#python train_model_important_counties_corn.py 2>&1 | tee ~/cs231n-satellite-images-models/runs/run2__dropout-0.50__corn/train_log
python train_model_important_counties_linear.py 2>&1 | tee ~/cs231n-satellite-images-models/runs/run9__linear__soybean/train_log
sudo poweroff
