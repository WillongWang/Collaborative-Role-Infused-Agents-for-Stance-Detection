# COLA

The code is modified based on [COLA](https://github.com/tsinghua-fib-lab/COLA/tree/main), and the datasets in ```/SemEval``` are modified from [SemEval-2016 Task 6](https://alt.qcri.org/semeval2016/task6/index.php?id=data-and-tools).

This project performs stance analysis on tweets, which consists of three stages with seven collaborative LLM agents (linguistic expert, domain specialist, social media veteran, pro-view debater, anti-view debater, stance concluder).  
The agents, through their assigned roles, activate their embedded domain knowledge to conduct a multi-dimensional analysis and reasoning of the tweets, ultimately determining the final stance through debate:

![](https://github.com/WillongWang/Collaborative-Role-Infused-Agents-for-Stance-Detection/blob/main/1.png)  
![](https://github.com/WillongWang/Collaborative-Role-Infused-Agents-for-Stance-Detection/blob/main/2.png)  

## Run  

Modify the LLM and API key in `cola.py`:  
```
client = OpenAI(api_key="sk-xx", base_url="") # base_url="https://api.deepseek.com"
'''
client = AzureOpenAI(
api_key="", # your api key
api_version="", # replace
azure_endpoint="https://hkust.azure-api.net") # replace
'''
```  
```
python cola.py
```  
The code takes `/SemEval/small.csv` as an example input, which contains tweet, target, and stance. The domain specialist determines the relevant domain as prior knowledge based on the target.  

## Result  
Stance accuracy:  
| Model               | Accuracy     |
|---------------------|--------------|
| gpt-35-turbo        | 0.6667 (66.67%) |
| gpt-4o-mini         | 0.6667 (66.67%) |
| gpt-4o              | 0.7333 (73.33%) |
| deepseek-chat (V3)  | 0.8000 (80.00%) |

### AzureOpenAI API bug  
The linguistic expert agent gets stuck and fails to output when analyzing the last line of tweet in `small.csv`, but it works fine when analyzed separately in `error.py`.  
```
import pandas as pd
df = pd.read_csv("result_cola.csv")
last_row_value = df.iloc[-1]['Linguist Analysis']
print(last_row_value) # "Error"
```




