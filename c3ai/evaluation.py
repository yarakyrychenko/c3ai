from datasets import load_dataset
import torch, os, time
from transformers import AutoTokenizer, AutoModelForCausalLM
from torch.utils.data import DataLoader
from tqdm import tqdm

class Generator:
    def __init__(self, model_name, access_token, max_length=1024, batch_size=8, device=None):
        self.access_token = access_token
        self.max_length = max_length
        self.batch_size = batch_size

        if torch.cuda.get_device_capability()[0] >= 8:
            torch_dtype = torch.bfloat16
            attn_implementation = "flash_attention_2"
        else:
            torch_dtype = torch.float16
            attn_implementation = "eager"

        self.tokenizer = AutoTokenizer.from_pretrained(model_name, token=access_token, padding_side="left")
        self.model = AutoModelForCausalLM.from_pretrained(model_name, 
                                                        token=access_token, 
                                                        torch_dtype=torch_dtype, 
                                                        device_map="auto",
                                                        attn_implementation=attn_implementation, 
                                                        max_length=max_length)
        
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("Device: ",self.device)
        self.model.eval()
        
    def format_prompt(self, row):
        row["prompt"] = row["chosen"][0]["content"]
        return row
    
    def tokenize_function(self, dataset):
        messages = [self.tokenizer.apply_chat_template([{'role': 'user', 'content': prompt}], tokenize=False, add_generation_prompt=True) for prompt in dataset["prompt"]]
        return self.tokenizer(messages, return_tensors="pt", padding=True, truncation=True, max_length=self.max_length)
    
    def generate_responses(self, tokenized_dataset, max_new_tokens, top_p=0.9, temperature=0.6):

        columns_to_keep = ['input_ids', 'attention_mask']
        tokenized_dataset.set_format(type='torch', columns=columns_to_keep)
    
        dataloader = DataLoader(tokenized_dataset, batch_size=self.batch_size)
    
        responses = []
        for batch in tqdm(dataloader):
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)

            with torch.no_grad():

                outputs = self.model.generate(
                    input_ids=input_ids, 
                    attention_mask=attention_mask, 
                    return_dict_in_generate=True,
                    pad_token_id=self.tokenizer.eos_token_id, 
                    output_logits=False,
                    do_sample=True, 
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_p=top_p
                )

                generated_toks = outputs.sequences[:, input_ids.shape[-1]:]
                batch_responses = self.tokenizer.batch_decode(generated_toks, skip_special_tokens=True)
                responses.extend(batch_responses)

        print(responses)
        return responses

def generate(args):
    torch.cuda.empty_cache()

    print("### Starting generation ###")
    access_token = os.getenv('HF_ACCESS_TOKEN')
    
    print("Loading the model...")
    generator = Generator(args.model, access_token)

    print("Loading the dataset...")
    data = load_dataset('json', data_files=args.data_files)['train']
    data = data.add_column('index', list(range(len(data))))

    print("Generating prompts...")
    data_prompt = data.map(generator.format_prompt, remove_columns=data.column_names)
    tokenized_dataset = data_prompt.map(generator.tokenize_function, batched=True) 

    print("Generating responses...")
    start_time = time.time()

    responses = generator.generate_responses(tokenized_dataset,max_new_tokens=args["max_new_tokens"], top_p=args["top_p"], temperature=args["temperature"])

    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Time taken: {elapsed_time:.4f} seconds")

    print("Saving results...")
    data = data.add_column('model_response', responses)
    
    output_ds = f'/results/eval_{args["name"]}.jsonl'

    data.to_json(output_ds, orient='records', lines=True)
    print(f'Results saved to {output_ds}')

    print("### Ended generation ###")
