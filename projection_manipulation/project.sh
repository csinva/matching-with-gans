# celeba generated
PROJECT="python ../lib/stylegan2/project_images.py"
ALIGN="python ../lib/stylegan2/align_images.py"



# stylegan2 generated #############################
# $PROJECT ../../data/annotation-dataset-stylegan2/images/ ../../data_processed/stylegan2/generated_images_0/ --regularize_mean_deviation_weight 0

# $PROJECT ../../data/annotation-dataset-stylegan2/images/ ../../data_processed/stylegan2/generated_images_1/ --regularize_mean_deviation_weight 1

# $PROJECT ../../data/annotation-dataset-stylegan2/images/ ../../data_processed/stylegan2/generated_images_10000/ --regularize_mean_deviation_weight 10000




# celeba-hq projections #############################

# $PROJECT ../../data/celeba-hq/ims/ ../../data_processed/celeba-hq/generated_images_0/ --regularize_mean_deviation_weight 0

# $PROJECT ../../data/celeba-hq/ims/ ../../data_processed/celeba-hq/generated_images_0.01/ --regularize_mean_deviation_weight 0.01


$PROJECT ../data/celeba-hq/ims/ ../data_processed/celeba-hq/generated_images_0.1/ --regularize_mean_deviation_weight 0.1


# $PROJECT ../data/celeba-hq/ims/ ../data_processed/celeba-hq/generated_images_1/ --regularize_mean_deviation_weight 1

# $PROJECT ../../data/celeba-hq/ims/ ../../data_processed/celeba-hq/generated_images_10/ --regularize_mean_deviation_weight 10 # not really worth it

# $PROJECT ../../data/celeba-hq/ims/ ../../data_processed/celeba-hq/generated_images_10000/ --regularize_mean_deviation_weight 10000

# for i in 0 1 2 3 4 5 6 7
#do
#	nohup $PROJECT ../data/celeba-hq/ims/ ../data_processed/celeba-hq/generated_images_0.1/ --regularize_mean_deviation_weight 0.1 --start_num $((3000 + $i * 3000)) --end_num $((6000 + $i * 3000)) --gpu $i > "proj_$i.log" &
#done



# align personal images ###################################
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
