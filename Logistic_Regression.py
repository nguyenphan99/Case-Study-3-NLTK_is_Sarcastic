import pandas as pd
import numpy as np
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import re
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt

#đọc file
path = "D:/Document/Machine_Learning/CS3/Sarcasm_Headlines_Dataset.json"
dataset = pd.read_json(path, lines = True)

corpus = []
#26709 điểm dữ liệu
for i in range(0,26709):
	#thay ! bằng từ exclamation trong tập dataset['headline'][i]
	review = re.sub('!', ' exclamation', dataset['headline'][i])
	#thay ? bằng từ inquiry
	review = review.replace('?', ' inquiry')
	#tìm văn bản trích dẫn
	matches = re.findall(r'\'(.+?)\'',review)
	if matches:
		review += ' quotation'
	#thay kí tự từ a đến z bằng khoảng trống 
	review = re.sub('[^a-z]', ' ' ,review)
	review = review.split()
	#biến đổi về từ gốc
	ps = PorterStemmer()
	
	review = [ps.stem(word) for word in review if not word in stopwords.words('english')]
	
	review = ' '.join(review)
	corpus.append(review)
#lấy nhãn
y = dataset.iloc[:, 2]

feature_n = range(100,3000,100)
scores = []
for i in feature_n:
	#vector hóa data
	cv = CountVectorizer(max_features = i)
	X = cv.fit_transform(corpus).toarray()
	#chia dữ liệu thành train và test
	from sklearn.model_selection import train_test_split
	X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.3, random_state = 0)
	#logistic
	Classifier = LogisticRegression()
	Classifier.fit(X_train, y_train)

	y_prediction = Classifier.predict(X_test)
	print(len(X_train))
	#tạo ra ma trận confusion
	from sklearn.metrics import confusion_matrix
	cm = confusion_matrix(y_test, y_prediction)
	TP = cm[0][0]
	FP = cm[0][1]
	FN = cm[1][0]
	TN = cm[1][1]
	#tính toán tỉ lệ lỗi dựa vào ma trận confusion
	error_rate = (FP+FN)/(TP+FP+FN+TN)
	scores.append(error_rate)

optimal_n = feature_n[scores.index(min(scores))]
#lưu model
import pickle
filename = 'D:/Document/Machine_Learning/CS3/finalized_model.sav'
pickle.dump(Classifier, open(filename, 'wb'))

print("the optimal number of max vector is %d" % optimal_n+" with an error rate of %.3f" % min(scores))
plt.plot(feature_n,scores)
plt.xlabel("Number of Max vector")
plt.ylabel("Error rate")
plt.show()
	








