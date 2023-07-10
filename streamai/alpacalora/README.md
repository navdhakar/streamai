# 🌩️ Scrol-Alpaca-LoRA

- 🤗 **This is simplified cloud finetuning and deployable version of llama-7b, made for UI interfaces(unstable)!**
- Original Implementation [here](https://github.com/tloen/alpaca-lora)
- Documentation(WIP)

## Training

```bash
git clone https://github.com/corporaai/alpaca-lora.git &&
cd alpaca-lora &&
chmod +x start_job.sh &&
./start_job.sh scroltest 'https://firebasestorage.googleapis.com/v0/b/pdf-analysis-saas.appspot.com/o/Other%2Fdataset.json?alt=media&token=28abd658-a308-4050-b631-54bab9b63a6b' 'scrol_token'
```

## Deploy

- Automatic Deploy can only be done with scrol dashboard.

  for manual deploy use:

  ```bash
  python generate.py \
    --load_8bit \
    --base_model 'decapoda-research/llama-7b-hf' \
    --lora_weights ./scroltest
  ```
