# specify directories  ###################################
DIR_ORIG="./samples" # should point to directory of original images (png, jpg, jpeg)
DIR_ALIGNED="$DIR_ORIG/aligned"         # intermediate directories will be created
DIR_PROJECTED="$DIR_ORIG/projected"
DIR_MANIPULATED="$DIR_ORIG/manipulated"
DIR_INTERPOLATED="$DIR_ORIG/interpolated"

# commands to be run
ALIGN="python 00_align_images.py"
PROJECT="python 01_project_images.py"
MANIPULATE="python 02_manipulate.py"
INTERPOLATE="python 02_interpolate.py"
INTERPOLATE_GRID="python 02_interpolate_grid.py"

# echo "aligning..."
# $ALIGN $DIR_ORIG $DIR_ALIGNED
# echo "projecting..."
# $PROJECT $DIR_ALIGNED $DIR_PROJECTED --regularize_mean_deviation_weight 0.1
echo "manipulating..."
$MANIPULATE $DIR_PROJECTED $DIR_MANIPULATED
# echo "interpolating..."
# $INTERPOLATE_GRID $DIR_PROJECTED $DIR_INTERPOLATED
# $INTERPOLATE $DIR_PROJECTED $DIR_INTERPOLATED



# misc directories real ims #############################
# DIR_ORIG="../data_processed/pilot-our-images-large/PPP/"
