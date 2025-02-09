import torch, time, os
import pandas as pd
from datasets import load_dataset
from datasets import Dataset, DatasetDict

from c3ai.preferences.formatting import Formatter
from c3ai.preferences.generating import Generator, APIGenerator

class Preferences:
    def __init__(self, principles, path_to_prefs: str):
        if isinstance(principles, pd.DataFrame):
            self.principles = principles
        elif is_csv(principles):
            self.principles = pd.read_csv(principles)
        else:
            raise ValueError("Principles a must be a .csv file or a pd.DataFrame object")

        if path_to_prefs:
            self.path = path_to_prefs if is_json_or_jsonl(path_to_prefs) else ValueError("Path to preferences must exist and be a .json or .jsonl file")

        self.preferences = self.load() if path_to_prefs else None
        self.preferences_df = self.make_df() if self.preferences else None

        self.formatter = None
        self.generator = None
        self.params = None
    
    def load(self):
        preferences = load_dataset('json', data_files=self.path)['train'] 
        print(f'{len(preferences)} preferences loaded from {self.path}')
        return preferences

    def save(self, output_name: str):
        if not self.preferences:
            raise ValueError("No preferences generated yet")
        
        if self.params is None and output_name is None:
            raise ValueError("Specify output_name or add params to save preferences")

        params = self.params
        if not os.path.exists('results'):
            os.makedirs('results')
        if output_name is not None:
            output_ds = f'results/{output_name}.jsonl' if '.jsonl' not in output_name else f'results/{output_name}'
        else:
            output_ds = f'results/{params["model_name"].split("/")[-1]}_{params["principle_name"]}_{params["data_name"]}.jsonl'

        self.formatter = Formatter(principles_csv=self.principles, few_shots=params["few_shots"], selected=params["statement_ids"]) if self.formatter is None else self.formatter

        preferences = self.formatter.save_formatted_response_dataset(self.preferences,output_ds_name=output_ds)
        
    def make_df(self):
        pref_df = self.preferences.to_pandas().drop(['chosen', 'rejected','convo'], axis=1) 
        merged_df = pref_df.merge(self.principles, left_on='principle_id', right_on='id', suffixes=('', '_y'))
        merged_df = merged_df.drop([col for col in merged_df.columns if col.endswith('_y') and col != 'principle_y'], axis=1)
        merged_df = merged_df.rename(columns={'principle_y': 'principle'})

        return merged_df

    def generate(self, dataset, params : dict):
        self.params = params

        if isinstance(dataset, (Dataset, DatasetDict)):
            data = dataset
        elif is_json_or_jsonl(dataset):
            data = load_dataset('json', data_files=dataset)['train'] 
        else:
            raise ValueError("For preference generation, data must be a .json or .jsonl file or a Dataset object")

        self.formatter = Formatter(principles_csv=self.principles, few_shots=params["few_shots"], selected=params["statement_ids"])

        if "openai" in params["model_name"]:
            self.generator = APIGenerator(params["model_name"], params["access_token"], max_length=params["max_length"])
        else:
            self.generator = Generator(params["model_name"], params["access_token"], chat=params["chat"], max_length=params["max_length"])

        print("### Starting preference generation ###")
        preferences = data.map(
            lambda batch: self.formatter.format_comparisons(
                batch=batch,
                sample_one=params["sample_one"]), 
            remove_columns=data.column_names, 
            batched=True, 
            batch_size=8
            )

        print(f"Generating {len(preferences)} responses...")
        start_time = time.time()

        if "openai" in params["model_name"]:
            responses, comparioson_results = self.generator.generate_responses(
                preferences, 
                params["max_new_tokens"], 
                params["temperature"]
                )
        else:
            tokenized_dataset = preferences.map(self.generator.tokenize_function, batched=True) 
            responses, comparioson_results = self.generator.generate_responses(
                        tokenized_dataset,
                        max_new_tokens=params["max_new_tokens"], 
                        top_p=params["top_p"], 
                        temperature=params["temperature"]
                        )

        prob_df = pd.DataFrame(comparioson_results)

        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"Time taken: {elapsed_time:.4f} seconds")

        # Add responses to preferences and format them
        preferences = preferences.add_column('response', responses)
        preferences = preferences.add_column('prob_A', list(prob_df['prob_A']))
        preferences = preferences.add_column('prob_B', list(prob_df['prob_B']))
        preferences = preferences.add_column('uncertainty', list(prob_df['uncertainty']))

        self.preferences = preferences
        
        self.preferences_df = self.make_df()

        print("### Ended preference generation ###")
        
def is_json_or_jsonl(path):
    return isinstance(path, str) and os.path.exists(path) and path.lower().endswith(('.json', '.jsonl'))

def is_csv(path):
    return isinstance(path, str) and os.path.exists(path) and path.lower().endswith('.csv')