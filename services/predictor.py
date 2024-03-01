import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel


class Predictor:
    MODEL_NAME = "bigscience/bloomz-3b"

    def __init__(self, model_path):
        self.bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )

        self.model = AutoModelForCausalLM.from_pretrained(
            self.MODEL_NAME,
            quantization_config=self.bnb_config,
            device_map="cuda:0",
            trus_remote_code=True,
        )

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.MODEL_NAME, add_bos_token=True, trust_remote_code=True
        )
        self.model = PeftModel.from_pretrained(self.model, model_path)

    def predict(self, prompt):
        inputs = self.tokenizer(prompt, return_tensors="pt").to("cuda")
        self.model.to("cuda")
        self.model.eval()
        
        with torch.no_grad():
            outputs = self.model.generate(**input, max_length=50, num_return_sequences=1)
            response =  self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            #Clean up the memory
            del inputs
            del outputs
            del self.model
            del self.tokenizer
            torch.cuda.empty_cache()
            
            return response
