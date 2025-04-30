import os, time, torch, numpy as np
from datasets import load_dataset
from peft import LoraConfig, PeftModel, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl import ORPOConfig, ORPOTrainer, setup_chat_format

def ORPO(args, orpo_args):
    torch.cuda.empty_cache()
    access_token = os.getenv('HF_ACCESS_TOKEN')

    path = args['path']
    base_model = args['base_model']
    new_model = args['new_model']

    if torch.cuda.get_device_capability()[0] >= 8:
        torch_dtype = torch.bfloat16
        attn_implementation = "flash_attention_2"
    else:
        torch_dtype = torch.float16
        attn_implementation = "eager"
    print('Attention: ', attn_implementation)

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch_dtype,
        bnb_4bit_use_double_quant=True,
    )

    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=['up_proj', 'down_proj', 'gate_proj', 'k_proj', 'q_proj', 'v_proj', 'o_proj']
    )

    tokenizer = AutoTokenizer.from_pretrained(base_model, token=access_token)

    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        quantization_config=bnb_config,
        device_map="auto",
        attn_implementation=attn_implementation, 
        token=access_token
    )

    model, tokenizer = setup_chat_format(model, tokenizer)
    model = prepare_model_for_kbit_training(model)

    print('Tokenizer and model loaded.')

    dataset = format_training_dataset(args['data_files'],args['n_examples'])

    print('Dataset loaded.')

    trainer = ORPOTrainer(
        model=model,
        args=orpo_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        peft_config=peft_config,
        tokenizer=tokenizer,
    )
    
    print('Model ready to train.')
    start_time = time.time()

    trainer.train()
    print('Model trained.')
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Time taken: {elapsed_time:.4f} seconds.")

    path_to_model = path + 'model/'+ new_model
    trainer.save_model(path_to_model)
    print(f'Model saved to {path_to_model}.')

    tokenizer = AutoTokenizer.from_pretrained(base_model, token=access_token)
    fp16_model = AutoModelForCausalLM.from_pretrained(
        base_model,
        token=access_token,
        low_cpu_mem_usage=True,
        return_dict=True,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    fp16_model, tokenizer = setup_chat_format(fp16_model, tokenizer)

    model = PeftModel.from_pretrained(fp16_model, path_to_model)
    model = model.merge_and_unload()

    print('Pushing model to Huggingface')
    model.push_to_hub(new_model, use_temp_dir=True, token=access_token, private=True)
    tokenizer.push_to_hub(new_model, use_temp_dir=True, token=access_token, private=True)
    print('Model and tokenizer pushed to hub (private).')
    print("Done!")

    import gc
    del trainer, model
    gc.collect()
    gc.collect()
    torch.cuda.empty_cache()

def format_chat_template(row, tokenizer=tokenizer):
    row["prompt"] = row["chosen"][0]["content"]
    row["chosen"] = tokenizer.apply_chat_template(row["chosen"], tokenize=False)
    row["rejected"] = tokenizer.apply_chat_template(row["rejected"], tokenize=False)
    return row

def format_responses(row):
    if row['same_choice_probs'] == 1: 
        pass
    elif row['same_choice_probs'] == 0:
        chosen = row['chosen']
        rejected = row['rejected']
        row['chosen'] = rejected
        row['rejected'] = chosen 
    elif random.random() < 0.5:
        chosen = row['chosen']
        rejected = row['rejected']
        row['chosen'] = rejected
        row['rejected'] = chosen
    else:
        pass
    return row


def format_training_dataset(data_files,n_examples):
    random = np.random.default_rng(123)

    dataset = load_dataset('json', data_files=data_files)['train'].shuffle(seed=123)

    if n_examples > 0:
        dataset = dataset.select(range(n_examples))

    dataset = dataset.map(
                            format_chat_template,
                                num_proc= os.cpu_count(),
                                )
    if 'same_choice_probs' in dataset.column_names:
        print('Swapping responses...')
        dataset = dataset.map(
                            format_responses,
                                num_proc= os.cpu_count(),
                                )
        
    dataset = dataset.train_test_split(test_size=0.01)

    return dataset