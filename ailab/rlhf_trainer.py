import torch
from tqdm import tqdm
import pandas as pd
tqdm.pandas()
from trl import PPOTrainer, PPOConfig, AutoModelForCausalLMWithValueHead
from trl.core import LengthSampler
import random
from datasets import Dataset, load_dataset
from transformers import AutoModelForSequenceClassification,AutoTokenizer,TrainingArguments,BitsAndBytesConfig
from peft import PeftModel
import fire
from pyarrow.parquet import ParquetFile
import pyarrow as pa 

def main(
    load_8bit: bool = False,
    base_model: str = "/cpfs01/user/chenyukun/llama/7B",
    lora_weights: str = "/cpfs01/user/chenyukun/llama/save",
    min_length: int = 2,
    max_length: int = 200
):

    config = PPOConfig(learning_rate=1.41e-5)

    model = AutoModelForCausalLMWithValueHead.from_pretrained(
            base_model,
            load_in_8bit=load_8bit,
            torch_dtype=torch.float16,
            device_map="auto",
        )
    # model = PeftModel.from_pretrained(
    #         model,
    #         lora_weights,
    #         torch_dtype=torch.float16,
    #     )
    # model = model.merge_and_unload()
    
    # ref_model = AutoModelForCausalLMWithValueHead.from_pretrained(
    #         base_model,
    #         load_in_8bit=load_8bit,
    #         torch_dtype=torch.float16,
    #         device_map="auto",
    #     )
    # ref_model = PeftModel.from_pretrained(
    #         model,
    #         lora_weights,
    #         torch_dtype=torch.float16,
    #     )
    # ref_model = ref_model.merge_and_unload()
    ref_model = AutoModelForCausalLMWithValueHead.from_pretrained(
            base_model,
            load_in_8bit=load_8bit,
            torch_dtype=torch.float16,
            device_map="auto",
        )
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    tokenizer.pad_token = tokenizer.eos_token

    #load dataset, only keep question and response rows
    pf = ParquetFile("/cpfs01/user/chenyukun/OpenOrca/1M-GPT4-Augmented.parquet") 
    first_200_rows = next(pf.iter_batches(batch_size = 200)) 
    df = pa.Table.from_batches([first_200_rows]).to_pandas() 
    df = df[["question","response"]] 

    #Preprocess dataset
    def build_dataset(tokenizer,input_min_text_length=min_length, input_max_text_length=max_length):
        tokenizer = tokenizer
        tokenizer.pad_token = tokenizer.eos_token
        ds = Dataset.from_pandas(df)
        input_size = LengthSampler(input_min_text_length, input_max_text_length)
        def tokenize(sample):
            sample["input_ids"] = tokenizer.encode(sample["question"])[: input_size()]
            sample["query"] = tokenizer.decode(sample["input_ids"])
            return sample
        ds = ds.map(tokenize, batched=False)
        ds.set_format(type="torch")
        return ds

    dataset = build_dataset(tokenizer)
    def collator(data):
        return dict((key, [d[key] for d in data]) for key in data[0])


    ppo_trainer = PPOTrainer(config, model, ref_model, tokenizer, dataset=dataset, data_collator=collator)


    device = ppo_trainer.accelerator.device
    if ppo_trainer.accelerator.num_processes == 1:
        device = 0 if torch.cuda.is_available() else "cpu"


    #Load the reward model
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True, #double quantization
        bnb_4bit_quant_type='nf4' #normal float 4
    )
    
    rm_model_name = "/cpfs01/user/chenyukun/xlm-roberta-large" #Path of your model
    rm_model_trained = AutoModelForSequenceClassification.from_pretrained(
            rm_model_name, 
            quantization_config = quantization_config,
            num_labels=1 #regression task
    )
    rm_lora_weights = "/cpfs01/user/chenyukun/rewardmodel/checkpoint-500"
    rm_model_trained = PeftModel.from_pretrained(
            rm_model_trained,
            rm_lora_weights,
            quantization_config = quantization_config,
            # torch_dtype=torch.bfloat16,
    )
    rm_model_trained = rm_model_trained.merge_and_unload()
    rm_tokenizer_trained = AutoTokenizer.from_pretrained(rm_model_name)

    if rm_tokenizer_trained.pad_token is None:
        rm_tokenizer_trained.pad_token = rm_tokenizer_trained.eos_token
        rm_model_trained.config.pad_token_id = rm_model_trained.config.eos_token_id

    gen_kwargs = {"min_length": -1, "top_k": 0.0, "top_p": 1.0, "do_sample": True, "pad_token_id": tokenizer.eos_token_id}

    #PPO
    output_min_length = 4
    output_max_length = 16
    output_length_sampler = LengthSampler(output_min_length, output_max_length)


    generation_kwargs = {
        "min_length": -1,
        "top_k": 0.0,
        "top_p": 1.0,
        "do_sample": True,
        "pad_token_id": tokenizer.eos_token_id,
    }


    for epoch, batch in tqdm(enumerate(ppo_trainer.dataloader)):
        query_tensors = batch["input_ids"]

        response_tensors = []
        for query in query_tensors:
            gen_len = output_length_sampler()
            generation_kwargs["max_new_tokens"] = gen_len
            response = ppo_trainer.generate(query, **generation_kwargs)
            response_tensors.append(response.squeeze()[-gen_len:])
        batch["response"] = [tokenizer.decode(r.squeeze()) for r in response_tensors]

        text = [q + r for q, r in zip(batch["query"], batch["response"])]
        encoding = rm_tokenizer_trained(text, return_tensors="pt",padding='max_length',truncation=True)
        outputs = rm_model_trained(**encoding)
        rewards = [torch.tensor(i) for i in outputs.logits]

        stats = ppo_trainer.step(query_tensors, response_tensors, rewards)
        ppo_trainer.log_stats(stats, batch, rewards)
        
if __name__ == "__main__":
    fire.Fire(main)