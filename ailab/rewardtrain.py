from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments
)
from datasets import Dataset,load_dataset
from trl import RewardTrainer
import torch
from peft import LoraConfig

#Set Qlora configs
quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True, #double quantization
        bnb_4bit_quant_type='nf4' #normal float 4
)
#Select a base model whch we need to train for reward modeling.
model_name = "/cpfs01/user/chenyukun/xlm-roberta-large" #Path of your model
model = AutoModelForSequenceClassification.from_pretrained(
    model_name, 
    quantization_config = quantization_config,
    num_labels=1 #regression task
)
tokenizer = AutoTokenizer.from_pretrained(model_name)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = model.config.eos_token_id

def preprocess_function(examples):
    new_examples = {
        "input_ids_chosen": [],
        "attention_mask_chosen": [],
        "input_ids_rejected": [],
        "attention_mask_rejected": [],
    }
    for chosen, rejected in zip(examples["chosen"], examples["rejected"]):
        tokenized_j = tokenizer(chosen, padding='max_length', truncation=True, max_length=512)
        tokenized_k = tokenizer(rejected, padding='max_length', truncation=True, max_length=512)

        new_examples["input_ids_chosen"].append(tokenized_j["input_ids"])
        new_examples["attention_mask_chosen"].append(tokenized_j["attention_mask"])
        new_examples["input_ids_rejected"].append(tokenized_k["input_ids"])
        new_examples["attention_mask_rejected"].append(tokenized_k["attention_mask"])

    return new_examples

train_dataset = load_dataset("/cpfs01/user/chenyukun/jsonds", split="train")
eval_dataset = load_dataset("/cpfs01/user/chenyukun/jsonds", split="test")
# preprocess the dataset and filter out QAs that are longer than script_args.max_length
train_dataset = train_dataset.map(
    preprocess_function,
    batched=True
)
train_dataset = train_dataset.filter(
    lambda x: len(x["input_ids_chosen"]) <= 512
    and len(x["input_ids_rejected"]) <= 512
)

eval_dataset = eval_dataset.map(
    preprocess_function,
    batched=True
)
eval_dataset = eval_dataset.filter(
    lambda x: len(x["input_ids_chosen"]) <= 512
    and len(x["input_ids_rejected"]) <= 512
)

training_args = TrainingArguments(
    output_dir="/cpfs01/user/chenyukun/rewardmodel",
    per_device_train_batch_size=64,
    evaluation_strategy="steps",
    logging_steps=500,
    num_train_epochs =1,
    report_to="tensorboard",

)
    
peft_config = LoraConfig(
    r=4, 
    lora_alpha=16, 
    bias="none", 
    task_type="SEQ_CLS", 
    modules_to_save=["scores"]
)
    
trainer = RewardTrainer(
    model=model,
    args=training_args,
    tokenizer=tokenizer,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    peft_config=peft_config
)

trainer.train()