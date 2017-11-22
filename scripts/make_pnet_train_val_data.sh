#!/usr/bin/env bash

NEED_VAL_SET=1
if [[ "$NEED_VAL_SET" == "1" ]]
then
    python ./prepare_data/gen_pnet_val_data.py
fi
python ./prepare_data/gen_pnet_train_data.py