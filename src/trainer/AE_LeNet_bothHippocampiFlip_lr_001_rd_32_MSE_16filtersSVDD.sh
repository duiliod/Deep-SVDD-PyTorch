#!/bin/bash

#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --mem=16G
#SBATCH --job-name="AE_LeNet_bothHippocampiFlip_lr_001_rd_32_MSE_16filtersSVDD"
#SBATCH --output=AE_LeNet_bothHippocampiFlip_lr_001_rd_32_MSE_16filtersSVDD_%x.out
#SBATCH --gres=gpu:1
#SBATCH --qos=longrunning

# Usage: sbatch --job-name=THA AE_LeNet_bothHippocampiFlip_lr_001_rd_32_MSE_16filtersSVDD.sh THA
# or just: ./AE_LeNet_bothHippocampiFlip_lr_001_rd_32_MSE_16filtersSVDD.sh THA
# $1 (or $STRUCTURE env var) must be one of: THA, PAL, PUT, HIP, AMY
STRUCTURE="${1:-$STRUCTURE}"
if [ -z "$STRUCTURE" ]; then
  echo "ERROR: missing structure. Usage: $0 {THA|PAL|PUT|HIP|AMY}" >&2
  exit 1
fi

python train_SVDD.py --configAE '../../configs/AE_LeNet_bothHippocampiFlip_lr_001_rd_32_MSE_16filters.yml' --configSVDD '../../configs/AE_LeNet_bothHippocampiFlip_lr_001_rd_32_MSE_16filtersSVDD.yml' --structure "$STRUCTURE"
#python train_SVDD.py --no-pretrained --configAE '../../configs/AE_LeNet_bothHippocampiFlip_lr_001_rd_32_MSE_16filters.yml' --configSVDD '../../configs/AE_LeNet_bothHippocampiFlip_lr_001_rd_32_MSE_16filtersSVDD.yml' --structure "$STRUCTURE"

