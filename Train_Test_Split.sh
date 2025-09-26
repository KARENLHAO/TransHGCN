#!/bin/bash
# 这个脚本会自动运行 Train_Test_Split.py，不使用循环

PYTHON=python  # 如果你系统里用 python3，就改成 python3

# # STRING 网络
# $PYTHON Train_Test_Split.py --data hESC --net STRING --num 500  --ratio 0.67 --p_val 0.5
# $PYTHON Train_Test_Split.py --data hESC --net STRING --num 1000 --ratio 0.67 --p_val 0.5
# $PYTHON Train_Test_Split.py --data hHEP --net STRING --num 500  --ratio 0.67 --p_val 0.5
# $PYTHON Train_Test_Split.py --data hHEP --net STRING --num 1000 --ratio 0.67 --p_val 0.5
# $PYTHON Train_Test_Split.py --data mDC  --net STRING --num 500  --ratio 0.67 --p_val 0.5
# $PYTHON Train_Test_Split.py --data mDC  --net STRING --num 1000 --ratio 0.67 --p_val 0.5
# $PYTHON Train_Test_Split.py --data mESC --net STRING --num 500  --ratio 0.67 --p_val 0.5
# $PYTHON Train_Test_Split.py --data mESC --net STRING --num 1000 --ratio 0.67 --p_val 0.5
# $PYTHON Train_Test_Split.py --data mHSC-E --net STRING --num 500 --ratio 0.67 --p_val 0.5
# $PYTHON Train_Test_Split.py --data mHSC-E --net STRING --num 1000 --ratio 0.67 --p_val 0.5
# $PYTHON Train_Test_Split.py --data mHSC-GM --net STRING --num 500 --ratio 0.67 --p_val 0.5
# $PYTHON Train_Test_Split.py --data mHSC-GM --net STRING --num 1000 --ratio 0.67 --p_val 0.5
# $PYTHON Train_Test_Split.py --data mHSC-L --net STRING --num 500 --ratio 0.67 --p_val 0.5
# $PYTHON Train_Test_Split.py --data mHSC-L --net STRING --num 1000 --ratio 0.67 --p_val 0.5

# # Specific 网络
# $PYTHON Train_Test_Split.py --data hESC --net Specific --num 500  --ratio 0.67 --p_val 0.5
# $PYTHON Train_Test_Split.py --data hESC --net Specific --num 1000 --ratio 0.67 --p_val 0.5
# $PYTHON Train_Test_Split.py --data hHEP --net Specific --num 500  --ratio 0.67 --p_val 0.5
# $PYTHON Train_Test_Split.py --data hHEP --net Specific --num 1000 --ratio 0.67 --p_val 0.5
# $PYTHON Train_Test_Split.py --data mDC  --net Specific --num 500  --ratio 0.67 --p_val 0.5
# $PYTHON Train_Test_Split.py --data mDC  --net Specific --num 1000 --ratio 0.67 --p_val 0.5
# $PYTHON Train_Test_Split.py --data mESC --net Specific --num 500  --ratio 0.67 --p_val 0.5
# $PYTHON Train_Test_Split.py --data mESC --net Specific --num 1000 --ratio 0.67 --p_val 0.5
# $PYTHON Train_Test_Split.py --data mHSC-E --net Specific --num 500 --ratio 0.67 --p_val 0.5
# $PYTHON Train_Test_Split.py --data mHSC-E --net Specific --num 1000 --ratio 0.67 --p_val 0.5
# $PYTHON Train_Test_Split.py --data mHSC-GM --net Specific --num 500 --ratio 0.67 --p_val 0.5
# $PYTHON Train_Test_Split.py --data mHSC-GM --net Specific --num 1000 --ratio 0.67 --p_val 0.5
# $PYTHON Train_Test_Split.py --data mHSC-L --net Specific --num 500 --ratio 0.67 --p_val 0.5
# $PYTHON Train_Test_Split.py --data mHSC-L --net Specific --num 1000 --ratio 0.67 --p_val 0.5

# Non-Specific 网络
$PYTHON Train_Test_Split.py --data hESC --net Non-Specific --num 500  --ratio 0.67 --p_val 0.5
$PYTHON Train_Test_Split.py --data hESC --net Non-Specific --num 1000 --ratio 0.67 --p_val 0.5
$PYTHON Train_Test_Split.py --data hHEP --net Non-Specific --num 500  --ratio 0.67 --p_val 0.5
$PYTHON Train_Test_Split.py --data hHEP --net Non-Specific --num 1000 --ratio 0.67 --p_val 0.5
$PYTHON Train_Test_Split.py --data mDC  --net Non-Specific --num 500  --ratio 0.67 --p_val 0.5
$PYTHON Train_Test_Split.py --data mDC  --net Non-Specific --num 1000 --ratio 0.67 --p_val 0.5
$PYTHON Train_Test_Split.py --data mESC --net Non-Specific --num 500  --ratio 0.67 --p_val 0.5
$PYTHON Train_Test_Split.py --data mESC --net Non-Specific --num 1000 --ratio 0.67 --p_val 0.5
$PYTHON Train_Test_Split.py --data mHSC-E --net Non-Specific --num 500 --ratio 0.67 --p_val 0.5
$PYTHON Train_Test_Split.py --data mHSC-E --net Non-Specific --num 1000 --ratio 0.67 --p_val 0.5
$PYTHON Train_Test_Split.py --data mHSC-GM --net Non-Specific --num 500 --ratio 0.67 --p_val 0.5
$PYTHON Train_Test_Split.py --data mHSC-GM --net Non-Specific --num 1000 --ratio 0.67 --p_val 0.5
$PYTHON Train_Test_Split.py --data mHSC-L --net Non-Specific --num 500 --ratio 0.67 --p_val 0.5
$PYTHON Train_Test_Split.py --data mHSC-L --net Non-Specific --num 1000 --ratio 0.67 --p_val 0.5

# # Lofgof 网络
# $PYTHON Train_Test_Split.py --data mESC --net Lofgof --num 500  --ratio 0.67 --p_val 0.5
# $PYTHON Train_Test_Split.py --data mESC --net Lofgof --num 1000 --ratio 0.67 --p_val 0.5