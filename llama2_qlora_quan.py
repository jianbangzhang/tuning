# !/usr/bin/env python
# -*-coding:utf-8 -*-

"""
# File       : llama2_qlora_quan.py.py
# Time       ：2024/2/20 9:34
# Author     ：jianbang
# version    ：python 3.10
# company    : IFLYTEK Co.,Ltd.
# emil       : whdx072018@foxmail.com
# Description：
"""
from transformers import AutoTokenizer,AutoModelForCausalLM
from transformers import TrainingArguments,DataCollatorForSeq2Seq,Trainer
from datasets import Dataset
import argparse
import logging
import torch
from peft import LoraConfig, TaskType, get_peft_model
import os

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')


class DataProcess:
    def __init__(self,json_file,mode,tokenizer,max_length,train_test_split=0.1):
        self.json_file=json_file
        self.mode=mode
        self.tokenizer=tokenizer
        self.prompt="你是一个对话理解专家，你需要从单轮或多轮的客户与坐席对话中，根据最一轮对话理解客户的意图，并填充对应的槽位，并输出合法的json格式。对话如下：\n"
        self.eos_token_id=tokenizer.eos_token_id
        self.max_length=max_length
        self.train_ratio=train_test_split


    def load_from_json(self):
        dataset=Dataset.from_json(self.json_file)
        return dataset

    def process_fn(self,example):
        MAX_LENGTH = self.max_length
        instruction = self.tokenizer("\n".join([self.prompt, example["input"]]).strip() + "\n\n开始回答: ",add_special_tokens=False)
        response = self.tokenizer(example["target"], add_special_tokens=False)
        input_ids = instruction["input_ids"] + response["input_ids"] + [tokenizer.eos_token_id]
        attention_mask = instruction["attention_mask"] + response["attention_mask"] + [1]
        labels = [-100] * len(instruction["input_ids"]) + response["input_ids"] + [self.eos_token_id]
        if len(input_ids) > MAX_LENGTH:
            input_ids = input_ids[:MAX_LENGTH]
            attention_mask = attention_mask[:MAX_LENGTH]
            labels = labels[:MAX_LENGTH]
        logging.info("process data into features.")
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels
        }

    def pipeline(self):
        dataset=self.load_from_json()
        dataset = dataset.map(self.process_fn)
        dataset=dataset.train_test_split(train_size=self.train_ratio, seed=250)
        return dataset["train"],dataset["test"]





class CausalModel:
    def __init__(self,model_name_or_path,low_cpu_mem_usage=True,
                    device_map="auto",load_in_4bit=True,
                    bnb_4bit_use_double_quant=True):
        self.model_name_or_path=model_name_or_path
        self.low_cpu_mem_usage=low_cpu_mem_usage
        self.device_map=device_map
        self.load_in_4bit=load_in_4bit
        self.bnb_4bit_use_double_quant=bnb_4bit_use_double_quant

    def init_model(self):
        model = AutoModelForCausalLM.from_pretrained(
            self.model_name_or_path,
            low_cpu_mem_usage=self.low_cpu_mem_usage,
            torch_dtype=torch.half,
            device_map=self.device_map,
            load_in_4bit=self.load_in_4bit,
            bnb_4bit_compute_dtype=torch.half,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=self.bnb_4bit_use_double_quant)

        config = LoraConfig(task_type=TaskType.CAUSAL_LM,)
        peft_model = get_peft_model(model, config)
        return peft_model



class Train:
    def __init__(self,model,config,save_interval):
        self.model=model
        self.config=config
        self.save_interval=save_interval

    def train(self,traindataset):
        trainer = Trainer(
            model=model,
            args=self.config,
            train_dataset=traindataset,
            data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True),
        )
        trainer.train()

    def test(self):
        pass

    def predict(self):
        pass

    def save(self):
        pass


def usrAgrs():
    parser = argparse.ArgumentParser(description="the config from user.")
    parser.add_argument("model_name_or_path",type=str,default="hfl/chinese-llama-2-1.3b",help="decoder only model is required.")
    parser.add_argument("mode",type=str,default="train",required=True,help="train or inference")
    parser.add_argument("max_length",type=int,default=512,required=True,help="the length of input tokens.")
    parser.add_argument("json_file",type=str,required=True,help="the path of data,json format is required.")
    parser.add_argument("per_device_train_batch_size",type=int,default=1,help="batch size of per gpu when training.")
    parser.add_argument("output_dir",default="./checkpoints",type=str,help="the path of saved model after training.")
    parser.add_argument("gradient_accumulation_steps",default=8,type=int,help="the size of batch acculated step when training.")
    parser.add_argument("num_train_epochs",type=int,default=10,help="number of trainning epochs.")
    parser.add_argument("save_interval",type=int,default=5,help="the frequency of saving checkpoints of model.")
    return parser.parse_args()




if __name__ == '__main__':
    config=usrAgrs()
    model_name_or_path=config.model_name_or_path
    mode=config.mode
    max_length=config.max_length
    json_file=config.json_file
    per_device_train_batch_size=config.per_device_train_batch_size
    output_dir=config.output_dir
    gradient_accumulation_steps=config.gradient_accumulation_steps
    num_train_epochs=config.num_train_epochs
    save_interval=config.save_interval

    final_model="final_model"
    final_model_saved_path=os.path.join(output_dir, final_model)
    os.makedirs(output_dir,exist_ok=True)
    os.makedirs(final_model_saved_path,exist_ok=True)

    # tokenizer instance
    tokenizer=AutoTokenizer.from_pretrained(model_name_or_path)
    tokenizer.padding_side = "right"
    tokenizer.pad_token_id=2

    # dataset traindataset testdataset
    coffeeDataset=DataProcess(json_file,mode,tokenizer,max_length)
    traindatset,testdataset=coffeeDataset.pipeline()

    # modeling
    Model=CausalModel(model_name_or_path)
    model=Model.init_model()
    logging.info("modeling casual LLM.")
    model.enable_input_require_grads()
    logging.info(model.print_trainable_parameters())

    # training config
    train_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=per_device_train_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        logging_steps=10,
        num_train_epochs=num_train_epochs,
        gradient_checkpointing=True,
        optim="paged_adamw_32bit",
    )
    trainer=Train(model,train_args,save_interval)
    if mode=="train":
        trainer.train(traindatset)
        model = model.merge_and_unload()
        print(f"Last Saving the target model to {final_model_saved_path}")
        model.save_pretrained(final_model_saved_path)
        tokenizer.save_pretrained(final_model_saved_path)
    if mode=="test":
        trainer.test()
    if mode=="predict":
        trainer.predict()





