import numpy as np, pandas as pd
from datasets import load_dataset

class BaseFormatter:
    def __init__(self, few_shots, seed=123):
        self.random = np.random.default_rng(seed)
        if few_shots != None:
            self.few_shots = self.make_few_shots(load_dataset('json', data_files=few_shots)['train'])
        else:
            self.few_shots = ''

    def make_few_shots(self, data):
        text = []
        for i in range(len(data)):
            convo = self.make_convo(data['chosen'][i][:-1])
            chosen = 'Assistant: ' + data['chosen'][i][-1]['content']
            rejected = 'Assistant: ' + data['rejected'][i][-1]['content']
            response_a = chosen if data['response_a'][i] == 'chosen' else rejected
            response_b = chosen if data['response_a'][i] != 'chosen' else rejected
            answer = 'A' if data['response_a'][i] == 'chosen' else 'B'
            principle = data['principle'][i]
            text.append(f'''Consider the following conversation:

{convo}

{principle}
Options:
A. {response_a}
B. {response_b}
Only answer A or B. The answer is: {answer}''')
        return '\n\n\n'.join(text) + '\n\n\n'

    def make_convo(self, convo_list):
        convo = []
        for item in convo_list:
            if item['role'] == 'user':
                convo.append('User: ' + item['content'])
            else:
                convo.append('Assistant: ' + item['content'])
        return '\n\n'.join(convo)
    
    def shorten_response(self, response, max_words=251):
        words = response.split()
        if len(words) > max_words:
            return ' '.join(words[:max_words]) + ' ...'
        return response

    def determine_same_choice(self, row):
        comparison = row['response']
        response_a = row['response_a']
        if 'A' in comparison and response_a == 'chosen':
            value = 1
        elif 'B' in comparison and response_a == 'rejected':
            value = 1
        elif 'A' in comparison and response_a == 'rejected':
            value = 0
        elif 'B' in comparison and response_a == 'chosen':
            value = 0
        else:
            value = -1
        row['same_choice'] = value 
        return row
    
    def determine_same_choice_probs(self, row):
        prob_a = row['prob_A']
        prob_b = row['prob_B']
        response_a = row['response_a']
        if prob_a > prob_b and response_a == 'chosen':
            value = 1
        elif prob_a < prob_b  and response_a == 'rejected':
            value = 1
        elif prob_a > prob_b  and response_a == 'rejected':
            value = 0
        elif prob_a < prob_b  and response_a == 'chosen':
            value = 0
        else:
            value = -1
        row['same_choice_probs'] = value 
        return row
        
    def format_comparisons(self, batch):
        comparisons = {'comparison_prompt': [], 'response_a': [], 'principle_id': [], 
                       'convo': [], 'principle': [],
                        'response_a_text': [], 'response_b_text': [],
                       'prompt_id': [], 'chosen': [], 'rejected': []}

        for i in range(len(batch['chosen'])):
            chosen = self.shorten_response('Assistant: ' + batch['chosen'][i])
            rejected = self.shorten_response('Assistant: ' + batch['rejected'][i])
            convo = 'User: ' + batch['question'][i]

            principle = batch['principle'][i]
            principle_id = batch['principle_id'][i]
            
            for j in [0,1]:
                response_a = chosen if j==0 else rejected
                response_b = rejected if j==0 else chosen

                text = self.few_shots + f'''Consider the following conversation:

{convo}

{principle}
Options:
A. {response_a}
B. {response_b}
Only answer A or B. The answer is:'''
            
                comparisons['comparison_prompt'].append(text)
                comparisons['response_a'].append('chosen' if j==0 else 'rejected')
                comparisons['principle_id'].append(principle_id)
                comparisons['convo'].append(convo.replace('User: ','').replace('Assistant: ','')) 
                comparisons['principle'].append(principle)
                comparisons['response_a_text'].append(response_a.replace('Assistant: ',''))
                comparisons['response_b_text'].append(response_b.replace('Assistant: ',''))
                comparisons['prompt_id'].append(batch['index'][i])
                comparisons['chosen'].append(batch['chosen'][i])
                comparisons['rejected'].append(batch['rejected'][i])

        return comparisons
    
    def format_comparisons_half(self, batch):
        comparisons = {'comparison_prompt': [], 'response_a': [], 'principle_id': [], 
                       'convo': [], 'principle': [],
                        'response_a_text': [], 'response_b_text': [],
                       'prompt_id': [], 'chosen': [], 'rejected': []}

        for i in range(len(batch['chosen'])):
            chosen = self.shorten_response('Assistant: ' + batch['chosen'][i])
            rejected = self.shorten_response('Assistant: ' + batch['rejected'][i])
            convo = 'User: ' + batch['question'][i]

            principle = batch['principle'][i]
            principle_id = batch['principle_id'][i]
            
            if True:
                j = 1
                response_a = chosen if j==0 else rejected
                response_b = rejected if j==0 else chosen

                text = self.few_shots + f'''Consider the following conversation:

{convo}

{principle}
Options:
A. {response_a}
B. {response_b}
Only answer A or B. The answer is:'''
            
                comparisons['comparison_prompt'].append(text)
                comparisons['response_a'].append('chosen' if j==0 else 'rejected')
                comparisons['principle_id'].append(principle_id)
                comparisons['convo'].append(convo.replace('User: ','').replace('Assistant: ','')) 
                comparisons['principle'].append(principle)
                comparisons['response_a_text'].append(response_a.replace('Assistant: ',''))
                comparisons['response_b_text'].append(response_b.replace('Assistant: ',''))
                comparisons['prompt_id'].append(batch['index'][i])
                comparisons['chosen'].append(batch['chosen'][i])
                comparisons['rejected'].append(batch['rejected'][i])

        return comparisons
        
class Formatter(BaseFormatter):
    def __init__(self, principles_csv, few_shots, selected=None, seed=123):
        super().__init__(few_shots,seed)
        self.principles_df = pd.read_csv(principles_csv) if type(principles_csv) == str else principles_csv
        self.selected = selected
        if self.selected:
            if type(self.selected) == str:
                with open(self.selected, 'r') as file:
                    content = file.read()
                    self.principle_ids = list(content.split("\n"))
            elif type(self.selected) == list:
                self.principle_ids = self.selected
            else:
                raise ValueError('Selected must be a list of principle ids or a file path to a txt new-line-separated list of principle ids')
        else:
            self.principle_ids = list(self.principles_df['id'])

        print(f"Includes {len(self.principle_ids)} principles:")
        print(self.principle_ids)

    def format_comparisons(self, batch, sample_one):
        comparisons = {'comparison_prompt': [], 'response_a': [], 'principle_id': [], 
                       'convo': [], 'principle': [],
                        'response_a_text': [], 'response_b_text': [],
                       'prompt_id': [], 'chosen': [], 'rejected': []}

        n_sample = 1 if sample_one else len(self.principle_ids)

        for i in range(len(batch['chosen'])):
            chosen = self.shorten_response('Assistant: ' + batch['chosen'][i][-1]['content'])
            rejected = self.shorten_response('Assistant: ' + batch['rejected'][i][-1]['content'])
            
            randomization = self.random.random() < 0.5
            response_a = chosen if randomization else rejected
            response_b = rejected if randomization else chosen
            convo = self.make_convo(batch['chosen'][i][:-1])

            rows = self.random.choice(self.principle_ids, n_sample, replace=False)
            for row in rows:
                principle = self.principles_df[self.principles_df['id'] == row]['principle'].values[0]
                principle_id = row

                text = self.few_shots + f'''Consider the following conversation:

{convo}

{principle}
Options:
A. {response_a}
B. {response_b}
Only answer A or B. The answer is:'''
            
                comparisons['comparison_prompt'].append(text)
                comparisons['response_a'].append('chosen' if randomization else 'rejected')
                comparisons['principle_id'].append(principle_id)
                comparisons['convo'].append(convo.replace('User: ','').replace('Assistant: ','')) 
                comparisons['principle'].append(principle)
                comparisons['response_a_text'].append(response_a.replace('Assistant: ',''))
                comparisons['response_b_text'].append(response_b.replace('Assistant: ',''))
                comparisons['prompt_id'].append(batch['index'][i])
                comparisons['chosen'].append(batch['chosen'][i])
                comparisons['rejected'].append(batch['rejected'][i])

        return comparisons