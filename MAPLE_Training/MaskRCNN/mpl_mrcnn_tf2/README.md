The following sbathc script is shared to get an idea of comand line execution of the file. This script is to execute the traning on multi gpu tacc hpc. 

```
#!/bin/bash
#SBATCH -J mpTrnTF2
#SBATCH -o ../.out/mpl_trn_tf2_log.out
#SBATCH -e ../.out/mpl_trn_tf2_log.err
#SBATCH -p rtx
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 0:10:00
#SBATCH -A DP

source /home/user/.bashrc
module load hdf5
module unload impi
module load cuda/10.1 cudnn nccl
conda activate maple_tf2
CUDA_VISIBLE_DEVICES=0,1,2,3
python Maple_TrainTf2.py train --dataset=/dataset_00_to_06 --weights=coco --output=/newlogs --logs=/newlogs
```
