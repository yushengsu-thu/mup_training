cd config

echo "Please set your training config and choose the fsdp. If you do not know how to set it, please refer to the config/default_config_oracle.yaml (loading...)"
accelerate config
config_dir=$(find~ -name default_config.yaml)
cp $config_dir .

echo "Your training config:"
ls

cd ../script
bash distill_llm.sh
