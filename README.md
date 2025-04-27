# COLA

The code is modified based on [COLA](https://github.com/tsinghua-fib-lab/COLA/tree/main), and the datasets in ```/SemEval``` are modified from [SemEval-2016 Task 6](https://alt.qcri.org/semeval2016/task6/index.php?id=data-and-tools).

This project performs stance analysis on tweets, which consists of three stages with seven collaborative LLM agents (linguistic expert, domain specialist, social media veteran, pro-view debater, anti-view debater, stance concluder).  
The agents, through their assigned roles, activate their embedded domain knowledge to conduct a multi-dimensional analysis and reasoning of the tweets, ultimately determining the final stance through debate:

![](https://github.com/WillongWang/Collaborative-Role-Infused-Agents-for-Stance-Detection/blob/main/1.png)  
![](https://github.com/WillongWang/Collaborative-Role-Infused-Agents-for-Stance-Detection/blob/main/2.png)  
