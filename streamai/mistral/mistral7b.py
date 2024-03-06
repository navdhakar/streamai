from datasets import load_dataset
from accelerate import FullyShardedDataParallelPlugin, Accelerator
from torch.distributed.fsdp.fully_sharded_data_parallel import FullOptimStateDictConfig, FullStateDictConfig
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import prepare_model_for_kbit_training
from peft import LoraConfig, get_peft_model
import transformers
from datetime import datetime
from trl import SFTTrainer
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
# max_length = 512 # This was an appropriate max length for my dataset

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
    resume_checkpoint:str=None
):

    train_dataset = load_dataset('json', data_files=dataset_file, split='train[0:20%]')
    eval_dataset = load_dataset('json', data_files=dataset_file, split='train[20%:100%]')

    fsdp_plugin = FullyShardedDataParallelPlugin(
        state_dict_config=FullStateDictConfig(offload_to_cpu=True, rank0_only=False),
        optim_state_dict_config=FullOptimStateDictConfig(offload_to_cpu=True, rank0_only=False),
    )

    accelerator = Accelerator(fsdp_plugin=fsdp_plugin)
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )
    print(f"downloading base model {base_model}")
    print("please be patient downloading models can take some time")

    model = AutoModelForCausalLM.from_pretrained(base_model, quantization_config=bnb_config, device_map="auto")
    tokenizer = AutoTokenizer.from_pretrained(
    base_model,
    padding_side="left",
    add_eos_token=True,
    add_bos_token=True,
)
    tokenizer.pad_token = tokenizer.eos_token
    def generate_and_tokenize_prompt(prompt):
        return tokenizer(formatting_func(prompt),
        truncation=True,
        max_length=max_length,
        padding="max_length",
        )

    tokenized_train_dataset = train_dataset.map(generate_and_tokenize_prompt)
    tokenized_val_dataset = eval_dataset.map(generate_and_tokenize_prompt)
    model.gradient_checkpointing_enable()
    model = prepare_model_for_kbit_training(model)
    config = LoraConfig(
    r=32,
    lora_alpha=64,
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
    bias="none",
    lora_dropout=0.05,  # Conventional
    task_type="CAUSAL_LM",
)
    if(resume_checkpoint):
        print(f"resuming finetuning from checkpoint: {resume_checkpoint}")
        model = PeftModel.from_pretrained(model, resume_checkpoint)
        model = model.merge_and_unload()

    model = get_peft_model(model, config)
    print_trainable_parameters(model)
    if torch.cuda.device_count() > 1: # If more than 1 GPU
        model.is_parallelizable = True
        model.model_parallel = True
    model = accelerator.prepare_model(model)
    project = "journal-finetune"
    base_model_name = "mistral"

    trainer = SFTTrainer(
        model=model,
        train_dataset=tokenized_train_dataset,
        eval_dataset=tokenized_val_dataset,
        max_seq_length=max_length,
        formatting_func=formatting_func,
        args=transformers.TrainingArguments(
            output_dir=output_dir,
            warmup_steps=1,
            per_device_train_batch_size=2,
            gradient_accumulation_steps=1,
            gradient_checkpointing=False,
            num_train_epochs=num_train_epochs,
            # max_steps=-1,
            learning_rate=2.5e-5, # Want a small lr for finetuning
            bf16=True,
            optim="paged_adamw_8bit",
            logging_steps=25,              # When to start reporting loss
            logging_dir="./logs",        # Directory for storing logs
            save_strategy="steps",       # Save the model checkpoint every logging step
            save_steps=25,                # Save checkpoints every 50 steps
            evaluation_strategy="steps", # Evaluate the model every logging step
            eval_steps=25,               # Evaluate and save checkpoints every 50 steps
            do_eval=True,                # Perform evaluation at the end of training
        ),
        data_collator=transformers.DataCollatorForLanguageModeling(tokenizer, mlm=False),
        packing= True,
    )

    model.config.use_cache = False
    print("starting training")
    trainer.train()
    trainer.model.save_pretrained(output_dir)
if __name__ == "__main__":
    fire.Fire(train)
