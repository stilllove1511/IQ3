import os
import torch
import transformers
from datetime import datetime
from dotenv import load_dotenv
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import prepare_model_for_kbit_training, Loraconfig, get_perf_model
from accelerate import FullyShardedDataParallelPlugin, Accelerator
from torch.distributed.fsdp.fully_sharded_data_parallel import (
    FullOptimStatedictConfig,
    FullyShardedDataParallel,
)
from huggingface_hub import login

from services.dataset import *

load_dotenv()
hf_key = os.getenv("HUGGING_FACE_API_KEY")
login(token=hf_key)


class Trainer:
    BASE_MODEL_NAME = "bigscience/bloomz-3b"

    def __init__(self, upload_id, tran_dataset, validation_dataset):
        self.upload_id = upload_id
        self.bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )

        self.model = AutoModelForCausalLM.from_pretrained(
            self.BASE_MODEL_NAME, config=self.bnb_config
        )
        self.tokenizer = AutoTokenizer.from_pretrained("stabilityai/stablelm-zephyr-3b")
        self.tokenizer.pad_token = self.tokenizer.eos_token

        max_length = 10

        def generate_and_tokenize_prompt(prompt):
            result = self.tokenizer(
                formatting_func(prompt),
                truncation=True,
                max_length=max_length,
                padding="max_length",
            )
            result["labels"] = result["input_ids"].copy()
            return result

        self.tokenized_train_dataset = tran_dataset.map(generate_and_tokenize_prompt)
        self.tokenized_validation_dataset = validation_dataset.map(
            generate_and_tokenize_prompt
        )

    def setup_lora(self):
        self.model.gradient_checkpointing_enable()
        self.model = prepare_model_for_kbit_training(self.model)

        self.lora_cfg = Loraconfig(
            r=32,
            lora_alpha=64,
            target_modules=[
                "a_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
                "lm_head",
            ],
            bias="none",
            lora_dropout=0.05,
            task_type="Causal_LM",
        )

        self.model = get_perf_model(self.model, self.lora_cfg)

    def setup_accelerator(self):
        fsdp_plugin = FullyShardedDataParallelPlugin(
            state_dict_config=FullOptimStatedictConfig(
                offload_to_cpu=True, rank0_only=False
            ),
            optim_state_dict_config=FullOptimStatedictConfig(
                offload_to_cpu=True, rank0_only=False
            ),
        )

        accelerator = Accelerator(fsdp_plugin=fsdp_plugin)
        self.model = accelerator.prepare_model(self.model)

        if torch.cuda.device_count() > 1:
            self.model.is_parallelizable = True
            self.model.model_parallel = True

    def train(self, is_logged=False):
        if is_logged:
            import mlflow

            mlflow.start_run(run_name=f"Training for upload_id: {self.upload_id}")
            mlflow.log_params("upload_id", self.upload_id)
            mlflow.log_params("model_name", self.BASE_MODEL_NAME)
            mlflow.log_params("model_config", self.bnb_config)

        self.setup_lora()
        self.setup_accelerator()

        if torch.cuda.device_count() > 1:
            self.model.is_parallelizable = True
            self.model.model_parallel = True

        project = "custom-finetune"
        base_model_name = "bloomz-3b"
        run_name = base_model_name + "-" + project + f"-{self.upload_id}"
        output_dir = "./logs/" + run_name

        self.tokenizer.pad_token = self.tokenizer.eos_token

        self.trainer = transformers.Trainer(
            model=self.model,
            train_dataset=self.tokenized_train_dataset,
            eval_dataset=self.tokenized_validation_dataset,
            args=transformers.TrainingArguments(
                output_dir=output_dir,
                warmup_steps=1,
                per_device_train_batch_size=2,
                gradient_accumulation_steps=1,
                max_steps=500,
                learning_rate=2.5e-5,
                bf16=True,
                optim="paged_adaw_8bit",
                logging_dir="./logs",
                save_strategy="steps",
                eval_steps=50,
                do_eval=True,
                report_to="mlflow",
                run_name=f"{run_name}-{datetime.now().strftime('%Y-%m-%d-%H-%M')}",
            ),
            data_collator=transformers.DataCollatorForLanguageModeling(
                self.tokenizer, mlm=False
            ),
        )

        self.model.config.use_cache = False
        self.trainer.train()

        # Release memory
        del self.model
        del self.tokenizer
        tourch.cuda.empty_cache()

        if is_logged:
            active_run = mlflow.active_run()
            run_id = active_run.info.run_id
            mlflow.end_run()

            os.rename(f"mlruns/0/{run_id}", f"mlruns/0/{self.upload_id}")
