from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, PeftModel
from transformers import TrainingArguments
from trl import SFTTrainer

from datasets import load_dataset
import torch
import fire

def formatting_func(sample):
  bos_token = "<s>"
  system_message = "[INST]"
  response = str(sample["output"])
  input = sample["instruction"]
  eos_token = "</s>"

  full_prompt = ""
  full_prompt += bos_token
  full_prompt += system_message
  full_prompt += "\n" + input
  full_prompt += "[/INST]"
  full_prompt += response
  full_prompt += eos_token

  return full_prompt

def generate_response(prompt, model):
  encoded_input = tokenizer(prompt,  return_tensors="pt", add_special_tokens=True)
  model_inputs = encoded_input.to('cuda')

  generated_ids = model.generate(**model_inputs,
                                 max_new_tokens=512,
                                 do_sample=True,
                                 pad_token_id=tokenizer.eos_token_id)

  decoded_output = tokenizer.batch_decode(generated_ids)

  return decoded_output[0].replace(prompt, "")


def print_trainable_parameters(model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
    )


def train(
    base_model:str="mistralai/Mistral-7B-v0.1",
    dataset_file:str="",
    output_dir:str="",
    num_train_epochs:int=5,
    max_length:int=512,
    resume_checkpoint:str=None,
    batch_size:int=None
):
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    def tokenize_prompts(prompt):
        return tokenizer(formatting_func(prompt),
        truncation=True,
        max_length=max_length,
        padding="max_length",
        )
    train_dataset = load_dataset('json', data_files=dataset_file, split='train[0:20%]')
    eval_dataset = load_dataset('json', data_files=dataset_file, split='train[20%:100%]')

    tokenized_train_dataset = train_dataset.map(tokenize_prompts)
    tokenized_val_dataset = eval_dataset.map(tokenize_prompts)

    nf4_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16
    )

    print(f"downloading base model {base_model}")
    print("please be patient downloading models can take some time")
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        device_map='auto',
        quantization_config=nf4_config,
        use_cache=False,
        attn_implementation="flash_attention_2"

    )
    tokenizer = AutoTokenizer.from_pretrained(base_model)

    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    peft_config = LoraConfig(
        lora_alpha=16,
        lora_dropout=0.1,
        r=64,
        bias="none",
            target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
            "lm_head",
        ],
        task_type="CAUSAL_LM"
    )
    if(resume_checkpoint):
        print(f"resuming finetuning from checkpoint: {resume_checkpoint}")
        model = PeftModel.from_pretrained(model, resume_checkpoint)
        model = model.merge_and_unload()

    model = prepare_model_for_kbit_training(model)
    model = get_peft_model(model, peft_config)

    print_trainable_parameters(model)

    if torch.cuda.device_count() > 1: # If more than 1 GPU
        print(torch.cuda.device_count())
        model.is_parallelizable = True
        model.model_parallel = True


    args = TrainingArguments(
        output_dir = output_dir,
        num_train_epochs=num_train_epochs,
        # max_steps = 1000, # comment out this line if you want to train in epochs
        per_device_train_batch_size = batch_size,
        warmup_steps = 0.03,
        logging_steps=10,
        save_strategy="epoch",
        #evaluation_strategy="epoch",
        evaluation_strategy="steps",
        eval_steps=10, # comment out this line if you want to evaluate at the end of each epoch
        learning_rate=2.5e-5,
        bf16=True,
        # lr_scheduler_type='constant',
    )

    trainer = SFTTrainer(
        model=model,
        peft_config=peft_config,
        max_seq_length=max_length,
        tokenizer=tokenizer,
        packing=True,
        formatting_func=formatting_func, # this will aplly the create_prompt mapping to all training and test dataset
        args=args,
        train_dataset=tokenized_train_dataset,
        eval_dataset=tokenized_val_dataset
    )
    print("starting training")
    trainer.train()
    trainer.model.save_pretrained(output_dir)
if __name__ == "__main__":
    fire.Fire(train)
