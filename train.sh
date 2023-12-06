python train.py -dataset UKDALE -trainDates "2015-01-01" "2016-01-01" -models "FCNdAE" "multiFCNdAE" "multiUNET" "dAE" "biLSTM" -seqLen 500 -epoch 100

python train.py -dataset HOSP -trainDates "2018-05-01" "2019-02-28" -models "biLSTM" "dAE" "FCNdAE" "multiFCNdAE" "UNET"  -seqLen 1440 -epoch 100