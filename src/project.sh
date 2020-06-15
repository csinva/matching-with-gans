# celeba generated
# python align_images.py ../../data/celeba-hq-guha/images ../../data_processed/celeba-hq/aligned_images
# python project_images.py ../../data/celeba-hq/images/ ../../data_processed/celeba-hq/generated_images/






# stylegan2 generated #############################
# python project_images.py ../../data/annotation-dataset-stylegan2/images/ ../../data_processed/stylegan2/generated_images_0/ --regularize_mean_deviation_weight 0

# python project_images.py ../../data/annotation-dataset-stylegan2/images/ ../../data_processed/stylegan2/generated_images_1/ --regularize_mean_deviation_weight 1

# python project_images.py ../../data/annotation-dataset-stylegan2/images/ ../../data_processed/stylegan2/generated_images_10000/ --regularize_mean_deviation_weight 10000




# celeba-hq generated #############################

# python project_images.py ../../data/celeba-hq/ims/ ../../data_processed/celeba-hq/generated_images_0/ --regularize_mean_deviation_weight 0

# python project_images.py ../../data/celeba-hq/ims/ ../../data_processed/celeba-hq/generated_images_0.01/ --regularize_mean_deviation_weight 0.01

python ../models/stylegan2encoder/project_images.py ../data/celeba-hq/ims/ ../data_processed/celeba-hq/generated_images_0.1/ --regularize_mean_deviation_weight 0.1

# python project_images.py ../../data/celeba-hq/ims/ ../../data_processed/celeba-hq/generated_images_1/ --regularize_mean_deviation_weight 1

# python project_images.py ../../data/celeba-hq/ims/ ../../data_processed/celeba-hq/generated_images_10/ --regularize_mean_deviation_weight 10 # not really worth it

# python project_images.py ../../data/celeba-hq/ims/ ../../data_processed/celeba-hq/generated_images_10000/ --regularize_mean_deviation_weight 10000







# project real ims #############################

# python ../models/stylegan2encoder/project_images.py ../data_processed/pilot-our-images-large/PPP/ ../data_processed/pilot-our-images-large/PPP_proj/ --regularize_mean_deviation_weight 0.1

# python ../models/stylegan2encoder/project_images.py ../data_processed/pilot-our-images-large/G2P2/ ../data_processed/pilot-our-images-large/G2P2_proj/ --regularize_mean_deviation_weight 0.1

# python ../models/stylegan2encoder/project_images.py ../data_processed/pilot-our-images-large/KFS/ ../data_processed/pilot-our-images-large/KFS_proj/ --regularize_mean_deviation_weight 0.1


# python project_images.py ../../data_processed/pilot-our-images/KFS/ ../../data_processed/pilot-our-images/KFS_proj/ --regularize_mean_deviation_weight 0.1

# python project_images.py ../../data_processed/pilot-our-images/PPP/ ../../data_processed/pilot-our-images/PPP_proj/ --regularize_mean_deviation_weight 0.1

# python project_images.py ../../data_processed/pilot-our-images/G2P2/ ../../data_processed/pilot-our-images/G2P2_proj_0/ --regularize_mean_deviation_weight 0

# python project_images.py ../../data_processed/pilot-our-images/KFS/ ../../data_processed/pilot-our-images/KFS_proj_0/ --regularize_mean_deviation_weight 0

# python project_images.py ../../data_processed/pilot-our-images/PPP/ ../../data_processed/pilot-our-images/PPP_proj_0/ --regularize_mean_deviation_weight 0