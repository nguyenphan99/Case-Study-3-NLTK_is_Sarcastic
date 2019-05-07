import pandas as pd
from nltk.corpus import stopwords
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
import sklearn.metrics
import string
import re


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


# vector hóa train set
def Train(train_X, train_y, test_X, test_y):
	# Khởi tạo model tf-idf
	tf_idf = TfidfVectorizer()

	# khởi tạo model cho từng loại phân loại
	KNN = KNeighborsClassifier()
	logist = LogisticRegression()
	DC = DecisionTreeClassifier()

	# Pipeline
	reg_test_clf_logist = Pipeline([('Tf*idf', tf_idf), ('reg', logist)])
	reg_test_clf_KNN = Pipeline([('Tf*idf', tf_idf), ('reg', KNN)])
	reg_test_clf_DC = Pipeline([('Tf*idf', tf_idf), ('reg', DC)])

	# train
	reg_test_clf_logist.fit(train_X, train_y)
	reg_test_clf_KNN.fit(train_X, train_y)
	reg_test_clf_DC.fit(train_X, train_y)


	# dự đoán data test
	reg_predicted_KNN = reg_test_clf_KNN.predict(test_X)
	reg_predicted_DC = reg_test_clf_DC.predict(test_X)
	reg_predicted_logist = reg_test_clf_logist.predict(test_X)

	return reg_predicted_KNN, reg_predicted_logist, reg_predicted_DC

def main():

	path = "D:/Document/Machine_Learning/CS3/Sarcasm_Headlines_Dataset.json"
	path_stopwords = "D:/Document/Machine_Learning/CS3/stopwords.txt"

	cleaned_df = ReadData(path)

	# stopwords = ReadStopwords(path_stopwords)
	stopwords = stopwords.words('english') + list(string.punctuation)

	Data = PreProcessing_Data(cleaned_df, stopwords)

	# chia train và test
	train_X = Data['headline'][ :20000]
	train_y = Data['is_sarcastic'][ :20000]
	test_X = Data['headline'][20000: ]
	test_y = Data['is_sarcastic'][20000: ]


	reg_predicted_KNN, reg_predicted_logist, reg_predicted_DC = Train(train_X, train_y, test_X, test_y)
	 
	# in ra f_score của dự đoán với nhãn thực
	print("Decision_Tree_F1_score:",sklearn.metrics.f1_score(test_y, reg_predicted_KNN, average = 'micro'))
	print("K-nearesNeighbor_F1_score:",sklearn.metrics.f1_score(test_y, reg_predicted_KNN, average = 'micro'))
	print("Logistic_F1_score:",sklearn.metrics.f1_score(test_y, reg_predicted_logist, average = 'micro'))

if __name__ == '__main__':
	main()