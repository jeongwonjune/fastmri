python train.py \
  -b 1 \
  -e 200 \
  -l 0.001 \
  -r 500 \
  -n 'Varnet' \
  -t '/home/Data/train/' \
  -v '/home/Data/val/'\
  --cascade 8 --chans 14 --sens_chans 5 \
  --aug_delay 4 --aug_strength 0.5\
  --load 1