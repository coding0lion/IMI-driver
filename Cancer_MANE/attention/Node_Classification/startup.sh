#!/bin/bash
source activate lion_py_env
python main_Node_Classification_MANE_Attention.py  --dimensions 64  --epochs 50 --nview 5 --cancer CHOL
wait
python main_Node_Classification_MANE_Attention.py  --dimensions 64  --epochs 50 --nview 5 --cancer COAD
wait
python main_Node_Classification_MANE_Attention.py  --dimensions 64  --epochs 50 --nview 5 --cancer DLBC
wait
python main_Node_Classification_MANE_Attention.py  --dimensions 64  --epochs 50 --nview 5 --cancer KICH
wait
python main_Node_Classification_MANE_Attention.py  --dimensions 64  --epochs 50 --nview 5 --cancer KIRC
wait
python main_Node_Classification_MANE_Attention.py  --dimensions 64  --epochs 50 --nview 5 --cancer KIRP
