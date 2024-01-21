!pip install nltk
!pip install textblob
!pip install wordcloud
import pandas as pd
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
from textblob import Word, TextBlob
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from nltk.sentiment import SentimentIntensityAnalyzer
from warnings import filterwarnings
from wordcloud import WordCloud

filterwarnings('ignore')
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 200)
pd.set_option('display.float_format', lambda x: '%.2f' % x)


#amazon.xlsx verisini okutuyoruz.
df = pd.read_excel("PROJELER/NLP/KOZMOS_Amazon_Duygu_Analizi/amazon.dataset.xlsx")

df.head()
df.info()
#Review değişkeni üzerinde;
#a. Tüm harfleri küçük harfe çevirelim
df["Review"] = df["Review"].str.lower()

#b. Noktalama işaretlerini çıkarıyoruz.
df["Review"] = df["Review"].str.replace('[^\w\s]', '', regex=True)

#c. Yorumlarda bulunan sayısal ifadeleri çıkarıyoruz.
df['Review'] = df['Review'].str.replace('\d', '', regex=True)

#d. Bilgi içermeyen kelimeleri (stopwords) veriden çıkarıyoruz.

nltk.download('stopwords')
sw = stopwords.words('english')
df['Review'] = df['Review'].apply(lambda x: " ".join(x for x in str(x).split() if x not in sw))

#e. 1000'den az geçen kelimeleri veriden çıkarıyoruz
sil = pd.Series(' '.join(df['Review']).split()).value_counts()[-1000:]
df['Review'] = df['Review'].apply(lambda x: " ".join(x for x in x.split() if x not in sil))

# Lemmatization
nltk.download('wordnet')
df['Review'] = df['Review'].apply(lambda x: " ".join([Word(word).lemmatize() for word in x.split()]))
df['Review'].head(10)

#Metin Görselleştirme

#Barplot görselleştirme işlemi
tf = df["Review"].apply(lambda x: pd.value_counts(x.split(" "))).sum(axis=0).reset_index()
tf.columns = ["words", "tf"]
tf[tf["tf"] > 500].plot.bar(x="words", y="tf")
plt.show()

# WordCloud görselleştirme işlemi

text = " ".join(i for i in df.Review)

wordcloud = WordCloud(max_font_size=50,
                      max_words=100,
                      background_color="black").generate(text)
plt.figure()
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.show()

wordcloud.to_file("wordcloud.png")

#Duygu Analizi
# Python içerisindeki NLTK paketinde tanımlanmış olan SentimentIntensityAnalyzer nesnesini oluşturalım.
nltk.download('vader_lexicon')
sia = SentimentIntensityAnalyzer()

# Adım 2: SentimentIntensityAnalyzer nesnesi ile polarite puanlarını inceleyelim

df["Review"][0:10].apply(lambda x: sia.polarity_scores(x))
df["Review"][0:10].apply(lambda x: sia.polarity_scores(x)["compound"])
df["Review"][0:10].apply(lambda x: "pos" if sia.polarity_scores(x)["compound"] > 0 else "neg")
df["Sentiment_Label"] = df["Review"].apply(lambda x: "pos" if sia.polarity_scores(x)["compound"] > 0 else "neg")
df.groupby("Sentiment_Label")["Star"].mean()   #  pos ve neg kırılımında ürünlere verilen yıldız sayısı ortalamalarına baktık.
# NOT: SentimentIntensityAnalyzer ile yorumları etiketleyerek, yorum sınıflandırma makine öğrenmesi modeli için bağımlı değişken
# oluşturulmuş oldu.

# Makine Öğrenmesine Hazırlık
# Bağımlı ve bağımsız değişkenlerimizi belirleyerek datayı train test olarak ayırınız.

train_x, test_x, train_y, test_y = train_test_split(df["Review"],
                                                    df["Sentiment_Label"],
                                                    random_state=42)
# Makine öğrenmesi modeline verileri verebilmemiz için temsil şekillerini sayısala çevirmemiz gerekmekte;

# TF-IDF Word Level
tf_idf_word_vectorizer = TfidfVectorizer().fit(train_x)
x_train_tf_idf_word = tf_idf_word_vectorizer.transform(train_x)
x_test_tf_idf_word = tf_idf_word_vectorizer.transform(test_x)

# Modelleme (Lojistik Regresyon)

log_model = LogisticRegression().fit(x_train_tf_idf_word, train_y)

#  Kurmuş olduğunmuz model ile tahmin işlemleri gerçekleştirelim.
# Predict fonksiyonu ile test datasını tahmin ederek kaydediyoruz.
# classification_report ile tahmin sonuçlarınızı raporlayıp gözlemleyelim.
# cross validation fonksiyonunu kullanarak ortalama accuracy değerini hesaplayalım.

y_pred = log_model.predict(x_test_tf_idf_word)

print(classification_report(y_pred, test_y))

cross_val_score(log_model, x_test_tf_idf_word, test_y, cv=5).mean()

# Veride bulunan yorumlardan rastgele seçerek modele soralım
# sample fonksiyonu ile "Review" değişkeni içerisinden örneklem seçerek yeni bir değere atıyoruz.
# Elde ettiğiniz örneklemi modelin tahmin edebilmesi için CountVectorizer ile vektörleştiriyoruz.
# Vektörleştirdiğiniz örneklemi fit ve transform işlemlerini yaparak kaydediniz.
# Kurmuş olduğunuz modele örneklemi vererek tahmin sonucunu kaydediyoruz.
# Örneklemi ve tahmin sonucunu ekrana yazdıryoruz.

random_review = pd.Series(df["Review"].sample(1).values)
yeni_yorum = CountVectorizer().fit(train_x).transform(random_review)
pred = log_model.predict(yeni_yorum)
print(f'Review:  {random_review[0]} \n Prediction: {pred}')

#  Modelleme (Random Forest)
# Lojistik regresyon modeli ile sonuçları karşılaştıralım.

rf_model = RandomForestClassifier().fit(x_train_tf_idf_word, train_y)
cross_val_score(rf_model, x_test_tf_idf_word, test_y, cv=5, n_jobs=-1).mean()

# Random Forest daha başarılı sonuç verdi