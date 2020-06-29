ALIGN="python 00_align_images.py"
PROJECT="python 01_project_images.py"
MANIPULATE="python 02_manipulate.py"

# align personal images (intermediate dirs will be created) ###################################
DIR_ORIG="./sample_projection/" # ../data/test/
DIR_ALIGNED="$DIR_ORIG/aligned"
DIR_PROJECTED="$DIR_ORIG/projected"
DIR_MANIPULATED="$DIR_ORIG/manipulated"

echo "aligning..."
$ALIGN $DIR_ORIG $DIR_ALIGNED
echo "projecting..."
$PROJECT $DIR_ALIGNED $DIR_PROJECTED --regularize_mean_deviation_weight 0.1
echo "manipulating..."
$MANIPULATE $DIR_PROJECTED $DIR_MANIPULATED


# project real ims #############################

# $PROJECT ../data_processed/pilot-our-images-large/PPP/ ../data_processed/pilot-our-images-large/PPP_proj/ --regularize_mean_deviation_weight 0.1

# $PROJECT ../data_processed/pilot-our-images-large/G2P2/ ../data_processed/pilot-our-images-large/G2P2_proj/ --regularize_mean_deviation_weight 0.1

# $PROJECT ../data_processed/pilot-our-images-large/KFS/ ../data_processed/pilot-our-images-large/KFS_proj/ --regularize_mean_deviation_weight 0.1


# $PROJECT ../../data_processed/pilot-our-images/KFS/ ../../data_processed/pilot-our-images/KFS_proj/ --regularize_mean_deviation_weight 0.1

# $PROJECT ../../data_processed/pilot-our-images/PPP/ ../../data_processed/pilot-our-images/PPP_proj/ --regularize_mean_deviation_weight 0.1

# $PROJECT ../../data_processed/pilot-our-images/G2P2/ ../../data_processed/pilot-our-images/G2P2_proj_0/ --regularize_mean_deviation_weight 0

# $PROJECT ../../data_processed/pilot-our-images/KFS/ ../../data_processed/pilot-our-images/KFS_proj_0/ --regularize_mean_deviation_weight 0

# $PROJECT ../../data_processed/pilot-our-images/PPP/ ../../data_processed/pilot-our-images/PPP_proj_0/ --regularize_mean_deviation_weight 0
