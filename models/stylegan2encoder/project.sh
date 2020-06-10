# celeba generated
# python align_images.py ../../data/celeba-hq/images ../../data_processed/celeba-hq/aligned_images
# python project_images.py ../../data/celeba-hq/images/ ../../data_processed/celeba-hq/generated_images/

# stylegan2 generated
# python project_images.py ../../data/annotation-dataset-stylegan2/images/ ../../data_processed/stylegan2/generated_images_0/ --regularize_mean_deviation_weight 0

# python project_images.py ../../data/annotation-dataset-stylegan2/images/ ../../data_processed/stylegan2/generated_images_10000/ --regularize_mean_deviation_weight 10000

# python project_images.py ../../data/annotation-dataset-stylegan2/images/ ../../data_processed/stylegan2/generated_images_1/ --regularize_mean_deviation_weight 1

python project_images.py ../../data/annotation-dataset-stylegan2/images/ ../../data_processed/stylegan2/generated_images_0.1/ --regularize_mean_deviation_weight 0.1