from pyspark import SparkContext
from sklearn.preprocessing import StandardScaler
from pprint import pprint
from scipy import stats
import numpy as np
import csv
import sys 

'''
@brief : this will do the linear_regression on our input feature
		 and try to find the best coeffecients to fit the data (X) and
		 the corresponding output (Y). 
		 last column is always the mortality rate
		 rest columns are input feature vector 
'''
def my_linear_regression(word, feat):
	Y = feat[:,-1]
	X_orig = feat[:,:-1] 

	bias = np.ones((X_orig.shape[0], 1))
	X = np.hstack((X_orig, bias))

	Y = Y.reshape((Y.shape[0],1))
	Y = np.asmatrix(Y)

	# linear regression closed form
	coeff = np.linalg.pinv(np.dot(X.T, X)) * np.dot(X.T, Y)
	y_pred = np.matmul(X, coeff)
	# return the coefficient of the norm_freq to test for our hypothesis
	return (word, coeff.item(0), y_pred, Y, X_orig)

'''
@ brief: this function takes in beta value vector and a tuple
		 containing word, predicted y, Actual y, and feature array
		 as input  and calculate the pValue from the calculated 
		 tvalue  
'''
def compute_pvalue(beta, numWords, tup):
	_ , y_pred, Y, X_orig = tup

	nrows = X_orig.shape[0]
	df = nrows - (X_orig.shape[1]+1)
	rss =  np.square(np.subtract(y_pred, Y)).sum(axis=0)
	sq_error = rss/df
	
	X = X_orig[:,0]
 	
 	# Since we already standardised the X , mean(X) = 0
	t_val = np.divide(beta, np.sqrt(sq_error/np.sum(np.square(X))))
	p_val = 1 - stats.t.cdf(abs(t_val), df=df)
	corrected_pval  = p_val[0][0]*numWords
	return corrected_pval


'''
@brief : this will standardize all the columns so that we can
		 pass these to the linear regressor to get some meaningful
		 results. 
'''
def do_standardization(word, feat):
	feat = list(feat)
	scaler = StandardScaler()
	feat_std = scaler.fit_transform(feat)
	return (word, feat_std)

def sort_and_print_output(rdd, flag, numWords):
	
	# Collect the max 20 +ve correlated words
	rdd_out = sc.parallelize(rdd.takeOrdered(20, key=lambda x: -x[0])) \
				.map(lambda x: (x[1][0], x[0], compute_pvalue(x[0], numWords, x[1])))

	if flag:
		suffix = "controlled for income"
		i = 3
	else:
		suffix = ""
		i = 1 

	print(i,". -- Top 20 positively correlated with hd mortality " + suffix)
	pprint(rdd_out.collect())

	# Collect the max 20 -ve correlated words
	rdd_out = sc.parallelize(rdd.takeOrdered(20, key=lambda x: x[0])) \
				.map(lambda x: (x[1][0], x[0], compute_pvalue(x[0], numWords, x[1])))

	print(i+1,". -- Top 20 negatively correlated with hd mortality " + suffix)
	pprint(rdd_out.collect())

def runTests(sc):
	# Read the file name 
	wordfreq_data = sys.argv[1]
	county_data = sys.argv[2]

	# read word freq data in an rdd excluding the header
	rdd1 = sc.textFile(wordfreq_data).mapPartitions(lambda line: csv.reader(line))
	header = rdd1.first()
	rdd1 = rdd1.filter(lambda line : line != header)
	#pprint(rdd1.collect())

	# Read the county data in a seperate rdd excluding the header
	rdd2 = sc.textFile(county_data).mapPartitions(lambda line: csv.reader(line))
	header = rdd2.first()
	rdd2 = rdd2.filter(lambda line : line != header)
	#pprint(rdd2.collect())

	# We map out the key values as the word and the values will be tuple containing the features as well as the norm frequency
	# for which we need to test our hypothesis.. Value = (norm_freq, log_income, heart_diease_mortality)
	
	rdd1 = rdd1.map(lambda col: (col[0],(col[1], col[3])))
	rdd2 = rdd2.map(lambda col: (col[0],(col[23], col[24]))) # Get three columns mainly income and hd mortality

	# Next we join the two rdd's on the  county id , map out the words with the feature tuples as values
	# group by words and then perform standardization of the feature columns. Finally we perform linear 
	# regression to get the beta values.

	# We first run without controlling for income
	rdd_grouped = rdd2.join(rdd1).map(lambda tup: (tup[1][1][0] , (tup[1][1][1], tup[1][0][0], tup[1][0][1]))) \
		      		  .groupByKey() \
		      		  .map(lambda record: do_standardization(record[0], record[1])).persist()

	# get total number of words over which we ran lin reg to correct  
	numWords = rdd_grouped.count()

	rdd = rdd_grouped.map(lambda std_record: my_linear_regression(std_record[0], np.column_stack((std_record[1][:,0] , std_record[1][:,2])))) \
		      	 	 .map(lambda x: (x[1], (x[0], x[2], x[3], x[4]))).persist()
	
	# Send flag as 0 to tell that this hypothesis test is done without controlling for income.
	sort_and_print_output(rdd, 0, numWords)

	# We next run with controlling for income
	rdd = rdd_grouped.map(lambda std_record: my_linear_regression(std_record[0], std_record[1])) \
		      		 .map(lambda x: (x[1], (x[0], x[2], x[3], x[4]))).persist()

	# Send flag as 1 to tell that this hypothesis test is done while controlling for income.
	# Including the income as the feat while doing linear regression will take the weights which 
	# would inherently control the final results for income.
	sort_and_print_output(rdd, 1, numWords)

# disable spark logs to avoid unecessary flodding 
# messages 
def disable_logs(sc):
	logger = sc._jvm.org.apache.log4j
	logger.LogManager.getLogger("org"). setLevel(logger.Level.ERROR)
	logger.LogManager.getLogger("akka").setLevel(logger.Level.ERROR)

if __name__ == "__main__":
	sc = SparkContext(appName="HW2-1")
	disable_logs(sc)
	runTests(sc)
