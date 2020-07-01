PROJECT="python 01_project_images.py"


# celeba-hq projections #############################

# $PROJECT ../../data/celeba-hq/ims/ ../../data_processed/celeba-hq/generated_images_0/ --regularize_mean_deviation_weight 0

# $PROJECT ../../data/celeba-hq/ims/ ../../data_processed/celeba-hq/generated_images_0.01/ --regularize_mean_deviation_weight 0.01

$PROJECT ../data/celeba-hq/ims/ ../data_processed/celeba-hq/generated_images_0.1/ --regularize_mean_deviation_weight 0.1 # best one

# $PROJECT ../data/celeba-hq/ims/ ../data_processed/celeba-hq/generated_images_1/ --regularize_mean_deviation_weight 1

# $PROJECT ../../data/celeba-hq/ims/ ../../data_processed/celeba-hq/generated_images_10/ --regularize_mean_deviation_weight 10 # not really worth it

# $PROJECT ../../data/celeba-hq/ims/ ../../data_processed/celeba-hq/generated_images_10000/ --regularize_mean_deviation_weight 10000

# loop to project over more in parallel
# for i in 0 1 2 3 4 5 6 7
#do
#	nohup $PROJECT ../data/celeba-hq/ims/ ../data_processed/celeba-hq/generated_images_0.1/ --regularize_mean_deviation_weight 0.1 --start_num $((3000 + $i * 3000)) --end_num $((6000 + $i * 3000)) --gpu $i > "proj_$i.log" &
#done



# stylegan2 generated #############################
# $PROJECT ../../data/annotation-dataset-stylegan2/images/ ../../data_processed/stylegan2/generated_images_0/ --regularize_mean_deviation_weight 0

# $PROJECT ../../data/annotation-dataset-stylegan2/images/ ../../data_processed/stylegan2/generated_images_1/ --regularize_mean_deviation_weight 1

# $PROJECT ../../data/annotation-dataset-stylegan2/images/ ../../data_processed/stylegan2/generated_images_10000/ --regularize_mean_deviation_weight 10000