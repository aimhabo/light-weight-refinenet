#!/bin/sh
PYTHONPATH=$(pwd):$PYTHONPATH python src/train_multi.py \
--enc multi
