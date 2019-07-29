# niki.ai


The src file contains the code for model prediction
-Firstly we read the data and we take only the Star rating and Text review ,then we remove all the null values
-we made a function clean_review which
			- takes only the alphabet and removes all other punctuations
			- lemmatizes the word
			- removes all the stopwords
- we take the sentences and vectorizes all words in the word_vect_dict
- we then add the sentences after passing it with clean_review function into our csv
- we take the max length so that each sentences can be padded with that much length which becomes easier for the model
-we create an embedded matrix for the model
-we create the model and the accuracy comes out it to be around 90

Then we read the test data and apply the same procedures and put it on the model and stores the output in prediction.csv
