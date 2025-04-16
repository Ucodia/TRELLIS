@echo off
call conda activate trellis
set ATTN_BACKEND=flash-attn
set SPCONV_ALGO=native
set XFORMERS_FORCE_DISABLE_TRITON=1
python ./app.py 