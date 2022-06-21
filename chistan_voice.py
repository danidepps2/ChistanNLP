import speech_recognition as sr
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB

# clear terminal
print("\033c")

# variables
x = []
y = []
question = ""

# voice recognition
r = sr.Recognizer()
with sr.Microphone() as source:
    print("Listening...")
    r.pause_threshold = 1
    audio = r.listen(source, phrase_time_limit=7)
    print("Recognizing...")
    query = r.recognize_google(audio, language="fa-IR")
    # write voice text detected to question
    question = query

# predict data for prediction (you can use just question variable for predict and write question for result.txt  But for better understanding, I divided them into two variables)
predict_data = [query]

# read data and split data to x and y
query = open('question.txt', 'r',encoding='utf-8').read().split("\n")
for data in query:
    dt = data.split(":")
    x.append(dt[0])
    y.append(dt[1])

# text preprocessing and convert to data (fit x data [word] for transform to Usable data [number] )
vect = CountVectorizer()
x_vect = vect.fit_transform(x)

# normalized data (Optional)
tfidf = TfidfTransformer()
x_train = tfidf.fit_transform(x_vect)

# create nlp model
model = MultinomialNB().fit(x_train, y)

# convert predict data to usable data for model
predict_data = vect.transform(predict_data)
predict_data = tfidf.transform(predict_data)

# predict my data
predict = model.predict(predict_data)

# write the question and result in to the txt file
open('result.txt', 'w',encoding='utf-8').write(str(str(question)+"\n"+predict[0]))

print("End")