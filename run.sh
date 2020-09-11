#!/bin/bash
#$ -cwd
#$ -l rt_F=4
#$ -l h_rt=00:20:00
#$ -N cosmoflow
#$ -j y
#$ -o $JOB_NAME.o$JOB_ID

# ======== Modules ========
source /etc/profile.d/modules.sh
source modules.sh

# ======== Main ===========
mpirun -np 4 \
    -H localhost:4 \
    -bind-to none -map-by slot \
    -mca pml ob1 -mca btl ^openib \
    -x NCCL_DEBUG=INFO -x LD_LIBRARY_PATH -x PATH \
    python3 main.py

