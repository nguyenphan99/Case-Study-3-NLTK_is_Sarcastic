import pandas as pd
import numpy as np
from nltk.corpus import stopwords
import re
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import GaussianNB
import matplotlib.pyplot as plt
import sklearn.metrics
import string
import re
from sklearn.metrics import confusion_matrix

#đọc file
def ReadData(path):
	# đọc file
	raw_df = pd.read_json(path, lines = True)
	return raw_df


def ReadStopwords(path_stopwords):
	stopwords = []
	# đọc file stopwords.txt
	with open(path_stopwords,'r') as file:
		while True:
			line = file.readline()
			_line = line.split('\n')
			if not line:
				break
			stopwords.append(_line[0])
	# thêm các ký tự đặc biệt vào stopwords
	stopwords = list(stopwords) + list(string.punctuation)
	return stopwords

def PreProcessing_Data(cleaned_df, stopwords):
	# loại bỏ cột article_link
	cleaned_df.pop('article_link')

	# loại bỏ những dữ liệu bị thiếu
	cleaned_df.dropna()

	# khởi tạo stopwords bằng thư viện
	# stop = stopwords.words('english') + list(string.punctuation)

	# xử lý cột headline với stopwords
	cleaned_df['headline'] = cleaned_df['headline'].apply(lambda s : ' '.join([re.sub(r'\W+', '', word.lower()) for word in s.split(' ') if word not in stopwords]))
	
	return cleaned_df




def Train(feature_n, Data):
	scores = []
	for i in feature_n:
		#vector hóa data
		cv = CountVectorizer(max_features = i)
		X = cv.fit_transform(Data['headline']).toarray()

		# chia train và test
		train_X = X[:][ :20000]
		train_y = Data['is_sarcastic'][ :20000]
		test_X = X[:][20000: ]
		test_y = Data['is_sarcastic'][20000: ]
		#Naive_Bayes
		Classifier = GaussianNB()
		Classifier.fit(train_X, train_y)

		y_prediction = Classifier.predict(test_X)
		#tạo ra ma trận confusion
		
		cm = confusion_matrix(test_y, y_prediction)
		TP = cm[0][0]
		FP = cm[0][1]
		FN = cm[1][0]
		TN = cm[1][1]
		#tính toán tỉ lệ lỗi dựa vào ma trận confusion
		error_rate = (FP+FN)/(TP+FP+FN+TN)

		scores.append(error_rate)

	return scores
def main():
	path = "D:/Document/Machine_Learning/CS3/Sarcasm_Headlines_Dataset.json"
	path_stopwords = "D:/Document/Machine_Learning/CS3/stopwords.txt"

	cleaned_df = ReadData(path)

	# stopwords = ReadStopwords(path_stopwords)
	stopword = stopwords.words('english') + list(string.punctuation)

	Data = PreProcessing_Data(cleaned_df, stopword)

	
	#random sô lượng feature từ 100 đến 3000 bước nhảy là 100
	feature_n = range(100,3000,100)

	#train model
	scores = Train(feature_n,Data )

	# feature n tối ưu nhất
	optimal_n = feature_n[scores.index(min(scores))]

	print("the optimal number of max vector is %d" % optimal_n+" with an error rate of %.3f" % min(scores))
	plt.plot(feature_n,scores)
	plt.xlabel("Number of Max vector")
	plt.ylabel("Error rate")
	plt.show()

if __name__ == '__main__':
	main()



