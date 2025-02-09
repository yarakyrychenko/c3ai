# UNDER DEVELOPMENT

def format_chat_data(dataset):
        dataset = dataset.map(format_chat_template,num_proc= os.cpu_count())

        if 'same_choice_probs' in dataset.column_names:
            print('Swapping responses to reflect LLM-evaluated preferences...')
            dataset = dataset.map(
                                format_responses,
                                    num_proc= os.cpu_count(),
                                    )

def format_chat_template(row):
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