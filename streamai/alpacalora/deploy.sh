lora_weights_dir=$1
python3 -m venv "finetune-alpaca"

source "finetune-alpaca/bin/activate" 
echo "venv activated"

pip install -r requirements.txt

cp finetune-alpaca/lib/python3.8/site-packages/bitsandbytes/libbitsandbytes_cuda117.so finetune-alpaca/lib/python3.8/site-packages/bitsandbytes/libbitsandbytes_cpu.so
echo "bnb cuda library modified"

python generate.py \
    --load_8bit \
    --base_model 'decapoda-research/llama-7b-hf' \
    --lora_weights $lora_weights_dir
