echo "Choose the config: $1"
cp $1 /home/yusheng.su/.cache/huggingface/accelerate/$1
mv /home/yusheng.su/.cache/huggingface/accelerate/$1 /home/yusheng.su/.cache/huggingface/accelerate/default_config.yaml
echo "The replaced config setting as follows:"
cat /home/yusheng.su/.cache/huggingface/accelerate/default_config.yaml

