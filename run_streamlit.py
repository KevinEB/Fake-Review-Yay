import re
import json
import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
import seaborn as sns
import pickle
import nltk
import csv

from collections import OrderedDict
from nltk.tokenize import RegexpTokenizer
from collections import OrderedDict
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression,SGDClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, recall_score, precision_score, f1_score, pairwise_distances
from sklearn.model_selection import train_test_split



def main():
    st.title("Welcome to Shopee Reviewer")
    st.subheader("Tolong masukkan url kesini")

main()

# url link is used manually here as for now
url = st.text_input("Ini Teks URL: ")
r = re.search(r"i\.(\d+)\.(\d+)", url)
shop_id, item_id = r[1], r[2]


ratings_url = "https://shopee.co.id/api/v2/item/get_ratings?filter=0&flag=1&itemid={item_id}&limit=20&offset={offset}&shopid={shop_id}&type=0"
offset = 0
d = {"username": [], "userid": [], "rating": [], "comment": [], "ctime": []}
while True:
    data = requests.get(
        ratings_url.format(shop_id=shop_id, item_id=item_id, offset=offset)).json()

    # uncomment this to print all data:
    # print(json.dumps(data, indent=4))

    i = 1
    for i, rating in enumerate(data["data"]["ratings"], 1):
        d["username"].append(rating["author_username"])
        d["userid"].append(rating["userid"])
        d["rating"].append(rating["rating_star"])
        d["comment"].append(rating["comment"])
        d["ctime"].append(rating["ctime"])

        print(rating["author_username"])
        print(rating["userid"])
        print(rating["rating_star"])
        print(rating["comment"])
        print(rating["ctime"])
        print("-" * 100)

    if i % 20:
        break

    offset += 20


df = pd.DataFrame(d)
print(df)
df.to_csv("data.csv", index=False)

df['comment'] = df.apply(lambda row: str(row['comment']).lower(), axis=1)

tokenizer = RegexpTokenizer(r'\w+')
df['comment'] = df['comment'].apply(lambda x: ' '.join(word for word in tokenizer.tokenize(x)))

df['review_length'] = df['comment'].apply(lambda x: len(x.split()))

df['date'] = pd.to_datetime(df['ctime'],unit='s').dt.date
df['time'] = pd.to_datetime(df['ctime'],unit='s').dt.time

mnr_df1 = df[['userid', 'date']].copy()
mnr_df2 = mnr_df1.groupby(by=['date', 'userid']).size().reset_index(name='mnr')
mnr_df2['mnr'] = mnr_df2['mnr'] / mnr_df2['mnr'].max()
df = df.merge(mnr_df2, on=['userid', 'date'], how='inner')

review_data = df
res = OrderedDict()

for row in review_data.iterrows():
    if row[1].userid in res:
        res[row[1].userid].append(row[1].comment)
    else:
        res [row[1].userid] = [row[1].comment]

individual_reviewer = [{'userid': k, 'comment': v} for k, v in res.items()]
df2 = dict()
df2['userid'] = pd.Series([])
df2['Maximum Content Similarity'] = pd.Series([])
vector = TfidfVectorizer(min_df=0)
count = -1
for reviewer_data in individual_reviewer:
    count = count + 1
    try:
        tfidf = vector.fit_transform(reviewer_data['comment'])
    except:
        pass
    cosine = 1 - pairwise_distances(tfidf, metric='cosine')

    np.fill_diagonal(cosine, -np.inf)
    max = cosine.max()

    # To handle reviewier with just 1 review
    if max == -np.inf:
        max = 0
    df2['userid'][count] = reviewer_data['userid']
    df2['Maximum Content Similarity'][count] = max

df3 = pd.DataFrame(df2, columns=['userid', 'Maximum Content Similarity'])

# left outer join on original datamatrix and cosine dataframe
df = pd.merge(review_data, df3, on="userid", how="left")
df.drop(index=np.where(pd.isnull(df))[0], axis=0, inplace=True)

with open ('logreg_pickle', 'rb') as rasengan:
    logreg = pickle.load(rasengan)

X = df[['review_length', 'mnr', 'Maximum Content Similarity', 'rating']]

y_pred = logreg.predict(X[['review_length', 'mnr', 'Maximum Content Similarity', 'rating']])

X['fakeornot'] = y_pred

fake = X.fakeornot.str.count("fake").sum()
original = X.fakeornot.str.count("original").sum()

print(fake)
print(original)

labels = ['Komentar Tidak Membantu', 'Komentar Membantu',]
filter_opini = [fake, original]
colors = ['orange', 'seagreen']
explode = (0.1, 0)  # explode 1st slice

fig1, ax1 = plt.subplots()
ax1.pie(filter_opini, explode=explode, labels=labels, colors=colors,
        autopct='%1.1f%%', shadow=True, startangle=140)
ax1.axis('equal')
st.pyplot(fig1)
