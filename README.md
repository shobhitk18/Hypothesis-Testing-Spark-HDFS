# Hypothesis-Testing-Spark-HDFS
Did Hypothesis testing in spark over a Distributed File System

Input: Your code should take two command line parameters:    
        1) word_data (path to word data csv in the same format as the provided data).    
        2) heart_disease_data (path to outcome data csv).    
        Example: spark-submit A2_P2_SparkHT_LAST_ID.py ‘hdfs:feat.1gram.msgs.cnty.16to16.csv’.     
                ‘Hdfs:feat.1gram.msgs.cnty.16to16.csv’.     
## Task Requirements:    
Your objective is to compute the relationship between each word’s relative frequency (group_norm) and heart disease mortality using standardized multivariate linear regression to control for median income of the community (you will also need to run linear regression without the controls). There are over 10k words, thus you will be running over 10k independent linear regressions. You must use Spark such that each of these correlations can be run in parallel -- organize the data such that each record countains all counties for a single word, and then use a map to compute the correlation values for each. You must choose how to handle the outcome and control data effectively. You must implement multiple linear regression yourself -- it is just a line or two of matrix operations.  Finally, you must compute p values for each of the top 20 most positively and negatively correlated words and apply the Bonferonni multi-test correction.All together, your code should run in less than 5 minutes on the provided data. Your solution should be scalable, such that one simply needs to add more nodes to the cluster to handle 10x or 100x the data size.      
Other than the above, you are free to design what you feel is the most efficient and effective solution. Based on feedback, the instructor may add or modify restrictions (in minor ways) up to 2 days before the submission.  


Output: Your code should output four lists of results. For each word, output the triple: (“word”, beta_value, multi-test corrected p-value).    
1) The top 20 word positively correlated with heart disease mortality. 
2) The top 20 word negatively correlated with heart disease mortality,
3) The top 20 words positively related to hd mortality, controlling for income.  
4) The top 20 words negatively related to hd mortality, controlling for income.    
