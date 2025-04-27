# openai migrate

import openai
from openai import OpenAI, AzureOpenAI
# client = OpenAI(api_key="sk-xx", base_url="https://api.deepseek.com")
client = AzureOpenAI(
api_key="", # your api key
api_version="", # replace
azure_endpoint="https://hkust.azure-api.net") # replace

# Your API key
# TODO: The 'openai.api_base' option isn't read in the client API. You will need to pass it when you instantiate the client, e.g. 'OpenAI(base_url="https://xiaoai.plus/v1")'
# openai.api_base = "https://xiaoai.plus/v1"

import pandas as pd
import time
import logging

#assign experts for target
target_role_map = {
    "Atheism": "theologian",
    "Climate Change is a Real Concern": "environmental scientist",
    "Feminist Movement": "sociologist",
    "Hillary Clinton": "political scientist",
    "Legalization of Abortion": "sociologist",
    "Donald Trump": "political scientist"
}

def load_csv_data(file_path):
    encodings = ['utf-8', 'latin1', 'ISO-8859-1']
    for enc in encodings:
        try:
            return pd.read_csv(file_path, encoding=enc, engine='python')
        except UnicodeDecodeError:
            continue
    raise ValueError(f"Unable to read {file_path} with any of the encodings: {', '.join(encodings)}")

def get_completion_with_role(role, instruction, content):
    max_retries = 10
    for i in range(max_retries):
        try:
            messages = [
                {"role": "system", "content": f"You are a {role}."},
                {"role": "user", "content": f"{instruction}\n{content}"}
            ]
            response = client.chat.completions.create(
                model="gpt-4o", # deepseek-chat
                # model = "deployment_name"
                messages=messages,
                temperature=0)
            print(role, response.choices[0].message.content)
            return response.choices[0].message.content
        except (openai.RateLimitError, openai.InternalServerError, openai.APIError, openai.APITimeoutError,openai.APIConnectionError,openai.BadRequestError,openai.AuthenticationError):
            if i < max_retries - 1:
                time.sleep(10)
            else:
                logging.error('Max retries reached for prompt: ' + f"{role}. {instruction}. {content}")
                return "Error"

def get_completion(prompt):
    max_retries = 100

    for i in range(max_retries):
        try:
            messages = [{"role": "user", "content": prompt}]
            response = client.chat.completions.create(
                model="gpt-4o", # deepseek-chat
                # model = "deployment_name"
                messages=messages,
                temperature=0)
            return response.choices[0].message.content
        except (openai.RateLimitError, openai.InternalServerError, openai.APIError, openai.APITimeoutError,openai.APIConnectionError,openai.BadRequestError,openai.AuthenticationError):
            if i < max_retries - 1:
                time.sleep(2)
            else:
                logging.error('Max retries reached for prompt: ' + prompt)
                return "Error"

def linguist_analysis(tweet):
    instruction = "Accurately and concisely explain the linguistic elements in the sentence and how these elements affect meaning, including grammatical structure, tense and inflection, virtual speech, rhetorical devices, lexical choices and so on. Do nothing else."
    return get_completion_with_role("linguist", instruction, tweet)

def expert_analysis(tweet, target):
    role = target_role_map.get(target, "expert")
    instruction = f"Accurately and concisely explain the key elements contained in the quote, such as characters, events, parties, religions, etc. Also explain their relationship with {target} (if exist). Do nothing else."
    return get_completion_with_role(role, instruction, tweet)

def user_analysis(tweet):
    instruction = "Analyze the following sentence, focusing on the content, hashtags, Internet slang and colloquialisms, emotional tone, implied meaning, and so on. Do nothing else."
    return get_completion_with_role("heavy social media user", instruction, tweet)

def stance_analysis(tweet, ling_response, expert_response, user_response, target, stance):
    role = target_role_map.get(target, "expert")
    return get_completion(f"'''{tweet}'''\n <<<{ling_response}>>>\n [[[{expert_response}]]]\n---{user_response}---\n\
                          You think the attitude behind the sentence surrounded by ''' ''' is {stance} of {target}. \
                          The content enclosed by <<< >>> represents linguistic analysis. The content within [[[ ]]] represents the analysis of a {role}. \
                          The content enclosed by --- ---  represents the analysis of a heavy social media user. Identify the top three pieces of evidence from these that best support your opinion and argue for your opinion.")


#This is an example of a prompt for the final judgement stage. Most of the time, this prompt can be used directly.
#For some targets, it needs to include a more detailed explanation of the task to achieve the performance reported in our paper.
#We believe that automating the addition of specific explanations and evaluation criteria for the task is a direction for future improvement.
#If you have any questions, feel free to discuss them with me!
def final_judgement(tweet, favor_response, against_response, target):
    judgement=get_completion(f"Determine whether the sentence is in favor of or against {target}, or is irrelevant to {target}.\n \
                             Sentence: {tweet}\nJudge this in relation to the following arguments:\n\
                                Arguments that the attitude is in favor: {favor_response}\n\
                                    Arguments that the attitude is against: {against_response}\n\
                                            Choose from:\n A: Against\nB: Favor\nC: Irrelevant\n Constraint: Answer with only the option above that is most accurate and nothing else.")
    print(judgement)
    return judgement


def add_predictions_sequential(data):
    results = []  # To store the results

    for index, row in data.iterrows():
        tweet = row['Tweet']
        target = row['Target']

        # Step 1: Linguist analysis
        ling_response = linguist_analysis(tweet)

        # Step 2: Expert analysis
        expert_response = expert_analysis(tweet, target)

        # Step 3: Heavy social media user analysis
        user_response = user_analysis(tweet)

        # Step 4: Debate
        favor_response = stance_analysis(tweet, ling_response, expert_response, user_response, target, "in favor")
        print('favor', favor_response)
        against_response = stance_analysis(tweet, ling_response, expert_response, user_response, target, "against")
        print('against', against_response)

        # Step 5: Final judgement
        final_response = final_judgement(tweet, favor_response, against_response, target)

        # Construct the result for the current tweet and add it to the results list
        result = {
            'Tweet': tweet,
            'Target': target,
            'Linguist Analysis': ling_response,
            'Expert Analysis': expert_response,
            'User Analysis': user_response,
            'In Favor': favor_response,
            'Against': against_response,
            'Final Judgement': final_response
        }
        results.append(result)

    # Update the original dataframe with the results
    for idx, res in enumerate(results):
        for key, value in res.items():
            data.at[idx, key] = value

# Load the data
data = load_csv_data("SemEval/small.csv")

# Apply the predictions in a sequential manner
add_predictions_sequential(data)

# Save the modified data back to a CSV
data.to_csv("result_cola.csv", index=False)


# Stance accuracy
def calculate_accuracy(stance_file, judgement_file):
    # Read the CSV files
    df_stance = pd.read_csv(stance_file)
    df_judgement = pd.read_csv(judgement_file)

    # Ensure the DataFrames have the same number of rows
    if len(df_stance) != len(df_judgement):
        raise ValueError("The input files must have the same number of rows.")

    correct = 0
    total = len(df_stance)

    for stance, judgement in zip(df_stance['Stance'], df_judgement['Final Judgement']):
        # Convert to strings (in case of NaN or non-string values)
        if str(judgement) == "A": # deepseek-chat's responses
            judgement = "A: Against"
        if str(judgement) == "B": # deepseek-chat's responses
            judgement = "B: Favor"
        if str(judgement) == "C": # deepseek-chat's responses
            judgement = "C: Irrelevant"
        stance_str = str(stance).lower()
        judgement_str = str(judgement).lower()

        # Check if stance is a substring of judgement (case-insensitive)
        if stance_str in judgement_str:
            correct += 1

    accuracy = correct / total
    return accuracy

stance_file = "SemEval/small.csv"
judgement_file = "result_cola.csv"
accuracy = calculate_accuracy(stance_file, judgement_file)
print(f"Accuracy: {accuracy:.4f} ({(accuracy * 100):.2f}%)")






