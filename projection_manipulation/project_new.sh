# celeba generated
ALIGN="python align_images.py"
PROJECT="python ../lib/stylegan2/project_images.py"

# align personal images (intermediate dirs will be created) ###################################
DIR_ORIG="../data/personal-images/"
DIR_ALIGNED="../data/personal-images/aligned"
DIR_MANIPULATED="../data/personal-images/manipulated"
# $ALIGN ../data/personal-images/ ../data/personal-images/aligned

# $PROJECT ../data/personal-images/aligned ../data_processed/personal-images --regularize_mean_deviation_weight 0.1



# project real ims #############################

# $PROJECT ../data_processed/pilot-our-images-large/PPP/ ../data_processed/pilot-our-images-large/PPP_proj/ --regularize_mean_deviation_weight 0.1

# $PROJECT ../data_processed/pilot-our-images-large/G2P2/ ../data_processed/pilot-our-images-large/G2P2_proj/ --regularize_mean_deviation_weight 0.1

# $PROJECT ../data_processed/pilot-our-images-large/KFS/ ../data_processed/pilot-our-images-large/KFS_proj/ --regularize_mean_deviation_weight 0.1


# $PROJECT ../../data_processed/pilot-our-images/KFS/ ../../data_processed/pilot-our-images/KFS_proj/ --regularize_mean_deviation_weight 0.1

# $PROJECT ../../data_processed/pilot-our-images/PPP/ ../../data_processed/pilot-our-images/PPP_proj/ --regularize_mean_deviation_weight 0.1

# $PROJECT ../../data_processed/pilot-our-images/G2P2/ ../../data_processed/pilot-our-images/G2P2_proj_0/ --regularize_mean_deviation_weight 0

# $PROJECT ../../data_processed/pilot-our-images/KFS/ ../../data_processed/pilot-our-images/KFS_proj_0/ --regularize_mean_deviation_weight 0

# $PROJECT ../../data_processed/pilot-our-images/PPP/ ../../data_processed/pilot-our-images/PPP_proj_0/ --regularize_mean_deviation_weight 0
