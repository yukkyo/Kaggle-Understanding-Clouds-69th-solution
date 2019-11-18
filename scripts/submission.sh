

# Public LB 0.6645
python submission.py --config configs/model013.yaml --kfolds 1 --use-best

# Local: 0.6520, Public LB: 0.6596
python submission.py --kfolds 1 --config configs/model015.yaml --use-best --thres-top 0.5 --thres-bottom 0.4 --min-contours 22755,16718,16718,16718

# Local: 0.640, Public LB: 0.6554
python submission.py --kfolds 1 --config configs/model016.yaml --use-best --thres-top 0.5 --thres-bottom 0.4 --min-contours 12800,16718,10113,16718

# Local: 0.6518, Public LB: 0.6554
python submission.py --kfolds 1 --config configs/model017.yaml --use-best --thres-top 0.5 --thres-bottom 0.4 --min-contours 12800,16718,10113,16718

# Local: 0.6445, Public LB: 0.6592
python submission.py --kfolds 1 --config configs/model026.yaml --use-best --thres-top 0.6 --thres-bottom 0.4 --min-contours 8192,8192,6770,16718

#
python submission.py --kfolds 1 --config configs/model028.yaml --use-best --thres-top 0.7 --thres-bottom 0.3 --min-contours 10000,10000,10000,10000


python submission.py --kfolds 12345 --config configs/model028.yaml --use-best --


# top 0.6, bottom 0.4 is good ?

# Submit command
kaggle c submit -c understanding_cloud_organization -f -m
kaggle c submit -c understanding_cloud_organization -f -m "Add elipse seresnext50, chenge thres"

# Model039
kaggle c submit -c understanding_cloud_organization -f ../output/model/model039/sub_model039_kfolds12345_top060_minarea7500-12500-7500-12500_bottom045_usebest.csv -m "Add rotate90 in aug&tta, msunet"

