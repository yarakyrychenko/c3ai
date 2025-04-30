# Some code inspired by PairS #
# https://github.com/cambridgeltl/PairS/blob/main/pairs/local_model.py
# Aligning with Human Judgement: The Role of Pairwise Preference in Large Language Model Evaluators
# Liu et al.; arXiv preprint arXiv:2403.16950

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
import numpy as np
import openai, os

class BaseGenerator:
    def __init__(self, model_name, max_length):
        self.model_name = model_name
        self.max_length = max_length

    def calculate_uncertainty(self,probablities):
        probs = np.array(probablities)
        entropy = -np.sum(probs * np.log2(probs, where=probs > 0))
        return float(entropy)

class Generator(BaseGenerator):
    def __init__(self, model_name, chat=False, max_length=4096, device=None):
        super().__init__(model_name, max_length)

        self.chat = chat

        access_token = os.getenv('HF_ACCESS_TOKEN')
        
        # Load model and tokenizer
        if self.chat:
            print("Loading chat model from Hugging Face")
            self.tokenizer = AutoTokenizer.from_pretrained(model_name, token=access_token,padding_side='left')
            self.model = AutoModelForCausalLM.from_pretrained(model_name, 
                                                          token=access_token, 
                                                          torch_dtype=torch.float16, 
                                                          device_map="auto",
                                                          max_length=max_length)
        
        else:
            print("Loading base model from Hugging Face")
            self.tokenizer = AutoTokenizer.from_pretrained(model_name, token=access_token,padding_side='left')
            terminators = [self.tokenizer.eos_token_id,self.tokenizer.convert_tokens_to_ids("<|eot_id|>")]
            self.model = AutoModelForCausalLM.from_pretrained(model_name, 
                                                          token=access_token, 
                                                          torch_dtype=torch.float16, 
                                                          device_map="auto",
                                                          do_sample=False,
                                                          eos_token_id=terminators,
                                                          max_length=max_length)
        
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("Device: ",self.device)
        self.model.eval()

        self.A_ids = self.tokenizer.convert_tokens_to_ids(['A','ĠA'])   
        self.B_ids = self.tokenizer.convert_tokens_to_ids(['B','ĠB'])   

    def tokenize_function(self, dataset):
        if self.chat:
            messages = [self.tokenizer.apply_chat_template([{'role': 'user', 'content': prompt}], tokenize=False, add_generation_prompt=True) for prompt in dataset["comparison_prompt"]]
            return self.tokenizer(messages, return_tensors="pt", padding=True, truncation=True, max_length=self.max_length)
        else:
            messages = dataset["comparison_prompt"]
            return self.tokenizer(messages, return_tensors="pt", padding=True, truncation=True, max_length=self.max_length)
    
    def generate_responses(self, tokenized_dataset, max_new_tokens, top_p=0.9, temperature=0.6, batch_size=8):
        torch.cuda.empty_cache()

        columns_to_keep = ['input_ids', 'attention_mask']
        tokenized_dataset.set_format(type='torch', columns=columns_to_keep)
        
        dataloader = DataLoader(tokenized_dataset, batch_size=batch_size)
        
        responses = []
        comparisons = []
        for batch in tqdm(dataloader):
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)

            with torch.no_grad():

                vocab_size = self.model.config.vocab_size
                assert torch.max(input_ids) < vocab_size, "Token ID exceeds vocab size!"

                outputs = self.model.generate(
                    input_ids=input_ids, 
                    attention_mask=attention_mask, 
                    return_dict_in_generate=True,
                    pad_token_id=self.tokenizer.eos_token_id, 
                    output_logits=True,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_p=top_p
                )

                generated_toks = outputs.sequences[:, input_ids.shape[-1]:]
                batch_responses = self.tokenizer.batch_decode(generated_toks, skip_special_tokens=True)
                responses.extend(batch_responses)
                batch_comparisons = self.get_comparisons(generated_toks, outputs)
                comparisons.extend(batch_comparisons)

        return responses, comparisons

    def get_comparisons(self, generated_toks, outputs):
        comparisons = []
        for idx in range(generated_toks.shape[0]):
            seq_logits = [logits[idx] for logits in outputs.logits]  
            result = self.extract_probs(generated_toks[idx], seq_logits)
            comparisons.append(result)
        return comparisons
    
    def extract_probs(self, generated_toks, logits):
        for idx, token_id in enumerate(generated_toks):
            if token_id in self.A_ids or token_id in self.B_ids:
                logit = logits[idx]
                probs = F.softmax(logit, dim=-1)
                prob_A = max([probs[a_id].item() for a_id in self.A_ids])
                prob_B = max([probs[b_id].item() for b_id in self.B_ids])
                uncertainty = self.calculate_uncertainty([prob_A, prob_B])
                
                results = {"prob_A": prob_A, "prob_B": prob_B, "uncertainty": uncertainty}
                return results
            
        print("Failed to extract probs")
        print(self.tokenizer.decode(generated_toks))
        return  {"prob_A": .5, "prob_B": .5, "uncertainty": 1}

class APIGenerator(BaseGenerator):
    def __init__(self, model_name, max_length=4096):
        super().__init__(model_name, max_length)
        self.client = openai.OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
        self.A_ids = ["A"]
        self.B_ids = ["B"]

    def get_response(self, prompt, max_new_tokens, temperature):
        response = self.client.chat.completions.create(
            model=self.model_name.split('/')[-1],
            temperature=temperature,
            max_tokens = max_new_tokens,
            logprobs=True,
            top_logprobs=8,
            messages=[
                {"role": "user", "content": prompt}
            ]
        )
        reply = response.choices[0].message.content
        logprobs = response.choices[0].logprobs.content[0].top_logprobs 
        return reply, {logprob.token: logprob.logprob for logprob in logprobs}

    def extract_probs(self, logprob_dict):
        prob_A = max((np.exp(logprob_dict[a_id]) for a_id in self.A_ids if a_id in logprob_dict), default=None)
        prob_B = max((np.exp(logprob_dict[b_id]) for b_id in self.B_ids if b_id in logprob_dict), default=None)

        if prob_A is None and prob_B is not None:
            prob_A = 0.0001 # set a default probability
        elif prob_B is None and prob_A is not None:
            prob_B = 0.0001
        elif prob_A is None and prob_B is None:
            print("Failed to extract probs")
            prob_A = prob_B = 0.5   

        return {"prob_A": prob_A, "prob_B": prob_B, "uncertainty": self.calculate_uncertainty([prob_A, prob_B])}


    def generate_responses(self, dataset, max_new_tokens, temperature=0.6):
        responses = []
        comparisons = []

        for i in tqdm(range(len(dataset['comparison_prompt']))):
            prompt = dataset['comparison_prompt'][i]
            response, logprob_dict = self.get_response(prompt, max_new_tokens, temperature)
            responses.append(response)
            comparisons.append(self.extract_probs(logprob_dict))

        return responses, comparisons
