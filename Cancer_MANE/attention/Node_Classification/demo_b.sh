#!/bin/bash
source activate lion_py_env
python main_Node_Classification_MANE_Attention.py  --dimensions 64  --epochs 50 --nview 5 --cancer LIHC
wait
python main_Node_Classification_MANE_Attention.py  --dimensions 64  --epochs 50 --nview 5 --cancer READ
wait
python main_Node_Classification_MANE_Attention.py  --dimensions 64  --epochs 50 --nview 5 --cancer SARC
wait
python main_Node_Classification_MANE_Attention.py  --dimensions 64  --epochs 50 --nview 5 --cancer SKCM
wait
python main_Node_Classification_MANE_Attention.py  --dimensions 64  --epochs 50 --nview 5 --cancer STAD
wait
python main_Node_Classification_MANE_Attention.py  --dimensions 64  --epochs 50 --nview 5 --cancer TGCT