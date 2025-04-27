# AzureOpenAI API bug: The linguistic expert agent gets stuck and fails to output when analyzing the last line of Tweet in small.csv, but it works fine when analyzed separately in error.py.

# ERROR:root:Max retries reached for prompt: linguist. Accurately and concisely explain the linguistic elements in the sentence and how these elements affect meaning, including grammatical structure, tense and inflection, virtual speech, rhetorical devices, lexical choices and so on. Do nothing else.. @ktumulty The same reasons people believe in the Bible, Koran, Talmud, etc. Ignorance. Lack of education. Poverty of intelligence. #SemST
messages = [
                {"role": "system", "content": "You are a linguist."},
                {"role": "user", "content": "Accurately and concisely explain the linguistic elements in the sentence and how these elements affect meaning, including grammatical structure, tense and inflection, virtual speech, rhetorical devices, lexical choices and so on.\n@ktumulty The same reasons people believe in the Bible, Koran, Talmud, etc. Ignorance. Lack of education. Poverty of intelligence. #SemST"}
            ]

import pandas as pd
df = pd.read_csv("result_cola.csv")
last_row_value = df.iloc[-1]['Linguist Analysis']
print(last_row_value) # "Error"

from openai import AzureOpenAI
client = AzureOpenAI(
api_key="", # your api key
api_version="", # replace
azure_endpoint="https://hkust.azure-api.net") # replace
response = client.chat.completions.create(
                model="gpt-4o", # gpt-4o-mini, gpt-35-turbo
                # model = "deployment_name"
                messages=messages,
                temperature=0)
print(response.choices[0].message.content)

