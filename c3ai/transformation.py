import pandas as pd
import time
import openai

STATEMENT_PROMPT = """You are an assistant tasked with rewriting sentences in an appropriate format to make them statements. 

Definitions:
- A value is a belief pertaining to desirable end states or modes of conduct, that transcends specific situations, guides selection or evaluation of behavior, people, and events.
- A statement is a unit of input that directs behavior in accordance with a value. An example of a statement is: "The assistant should not discriminate based on gender." 

Your task:
- Rewrite each sentence in the form "The assistant should..." For example, "The assistant needs to..." should be rewritten as "The assistant should..."
- Sentences that are questions or begin with "Choose" or "Please choose" should be rephrased to fit the "The assistant should..." format. For example, "Please choose the response that is most supportive of life, liberty, and personal security" becomes "The assistant should be supportive of life, liberty, and personal security."
- If a sentence refers to the user with pronouns like "he" or "him," rephrase accordingly. For example, "It is important to him not to be humiliated" becomes "The assistant should never humiliate the user."
- When a sentence expresses someone's values, rephrase it to imply that the assistant should uphold those values. For instance, "He values justice for everyone, even strangers" becomes "The assistant should treat everyone justly, even people the user doesn't know."
- Replace references to "AI" or "The AI" with "The assistant," and replace references to "the human" with "the user." For example, "The AI should be helpful to the human" becomes "The assistant should be helpful to the user."
- Maintain the original meaning and structure as closely as possible, but avoid using phrases like "choose the response that."

Based on the definitions and instructions above, rewrite this sentence into a statement. Your reply should start with "The assistant should ". The sentence is: [SENTENCE]"""

PRINCIPLE_PROMPT = """You are an assistant tasked with rewriting statements into principles. 

Definitions:
- A value is a belief pertaining to desirable end states or modes of conduct, that transcends specific situations, guides selection or evaluation of behavior, people, and events.
- A statement is a unit of input that directs behavior in accordance with a value. An example of a statement is: 'The assistant should not discriminate based on gender.' 
- A principle is a translation of a statement into a form useful for picking between two assistant responses to a user. A principle is stated as a prompt to pick one of two possible response options that align best with the specific statement. An example of a principle is 'Choose the response that least discriminates based on gender.'

Your task:
- Rewrite the statement to make it into a principle by refrasing the statement into the form 'Choose the response that...'. 
- Preserve the original structure and meaning of the statement as much as possible.

Based on the definitions and instructions above, rewrite this statement into a principle. Your reply should start with 'Choose the response that' The statement is: [SENTENCE]"""

def get_response(text, prompt, model):
    response = openai.chat.completions.create(
        model=model,
        messages=[
            {"role": "user", "content": prompt.replace("[SENTENCE]", text)},
        ]
    )
    reply = response.choices[0].message.content
    return reply

def generate(col, prompt, sleep_secs, model):
    """Generates statements or principles from a dataframe of items using the given prompt and OpenAI model."""
    if prompt == 'statement':
        prompt = STATEMENT_PROMPT
    elif prompt == 'principle':
        prompt = PRINCIPLE_PROMPT

    print("Generating...")
    results = []
    for i in range(len(col)):
        result = get_response(col[i], prompt, model)
        print(i)
        print(col[i])
        print(result)
        results.append(result)
        if sleep_secs != None:
            time.sleep(sleep_secs)   
    return results
