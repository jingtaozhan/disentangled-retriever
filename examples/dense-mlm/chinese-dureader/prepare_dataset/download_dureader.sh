set -e 

dataset_dir="./data/datasets/dureader"

mkdir -p $dataset_dir
cd $dataset_dir

wget -nc --no-check-certificate https://dataset-bj.cdn.bcebos.com/dureader/dureader_preprocessed.zip

unzip dureader_preprocessed.zip

echo "Done"