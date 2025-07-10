import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
import itertools

# === stop-word removal
import nltk
from nltk.corpus import stopwords

# === n-gram analysis
from nltk import ngrams

# === TF–IDF
from sklearn.feature_extraction.text import TfidfVectorizer

# === topic modeling
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation

# === sentiment analysis
nltk.download('vader_lexicon')
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# === entity extraction
import spacy

# === added for document clustering
from sklearn.cluster import KMeans                        # <<< ADDED
from sklearn.decomposition import TruncatedSVD            # <<< ADDED

# === added for UMAP visualization
import umap.umap_ as umap                                # <<< ADDED

st.set_page_config(page_title='Text Data EDA', layout='wide')
st.title('Exploratory Data Analysis for Text Reports')

@st.cache_data
def load_data():
    return pd.read_csv('/Users/harris/Documents/Capstone_Project/Capstone_Project/Data/TCGA_Reports.csv')

df = load_data()

# Preview
st.header('Dataset Preview')
st.dataframe(df.head())

# Shape & missing
st.subheader('Shape & Missing Values')
st.write(f'Dataset shape: {df.shape}')
st.write(df.isnull().sum())

# Length features
st.subheader('Character & Word Counts')
df['char_count'] = df['text'].astype(str).apply(len)
df['word_count'] = df['text'].astype(str).apply(lambda x: len(x.split()))
st.write(df[['char_count', 'word_count']].describe())

# Word count distribution
st.subheader('Distribution of Word Counts')
fig, ax = plt.subplots()
ax.hist(df['word_count'], bins=30)
ax.set_xlabel('Word Count')
ax.set_ylabel('Frequency')
ax.set_title('Distribution of Word Counts')
st.pyplot(fig)

# Top 20 words (raw)
st.subheader('Top 20 Most Frequent Words (Raw)')
all_words = list(itertools.chain.from_iterable(df['text'].str.lower().str.split().tolist()))
common = pd.DataFrame(Counter(all_words).most_common(20), columns=['word', 'count'])
st.dataframe(common)

# stop-word removal
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))
clean_words = [w for w in all_words if w.isalpha() and w not in stop_words]
st.subheader('Top 20 Words After Stop-word Removal')
st.dataframe(pd.DataFrame(Counter(clean_words).most_common(20), columns=['word','count']))

# n-grams
st.subheader('Top 20 Bigrams')
bigrams = list(ngrams(clean_words,2))
st.dataframe(pd.DataFrame([(' '.join(b),c) for b,c in Counter(bigrams).most_common(20)], columns=['bigram','count']))

st.subheader('Top 20 Trigrams')
trigrams = list(ngrams(clean_words,3))
st.dataframe(pd.DataFrame([(' '.join(t),c) for t,c in Counter(trigrams).most_common(20)], columns=['trigram','count']))

# TF–IDF
st.subheader('Top 20 Terms by TF–IDF')
vectorizer = TfidfVectorizer(stop_words='english')
tfidf = vectorizer.fit_transform(df['text'])
tfidf_df = pd.DataFrame({
    'term': vectorizer.get_feature_names_out(),
    'tfidf': tfidf.mean(axis=0).A1
}).sort_values('tfidf', ascending=False).head(20)
st.dataframe(tfidf_df)

# Topic modeling
# st.subheader('Latent Topics (LDA)')
# count_vect = CountVectorizer(stop_words='english')
# count_mat = count_vect.fit_transform(df['text'])
# lda = LatentDirichletAllocation(n_components=5, random_state=0)
# lda.fit(count_mat)
# terms = count_vect.get_feature_names_out()
# topics = []
# for comp in lda.components_:
#     topics.append(' '.join([terms[i] for i in comp.argsort()[-10:][::-1]]))
# st.dataframe(pd.DataFrame({'Topic': [f'Topic {i+1}' for i in range(5)], 'Keywords': topics}))

# Sentiment analysis
# st.subheader('Sentiment Distribution')
# sia = SentimentIntensityAnalyzer()
# df['sentiment'] = df['text'].apply(lambda x: sia.polarity_scores(str(x))['compound'])
# fig2, ax2 = plt.subplots()
# ax2.hist(df['sentiment'], bins=30)
# ax2.set_xlabel('Compound Sentiment Score')
# ax2.set_ylabel('Frequency')
# ax2.set_title('Sentiment Scores')
# st.pyplot(fig2)
# st.write(df['sentiment'].describe())

# Entity extraction
# st.subheader('Top 20 Named Entities')
# nlp = spacy.load("en_core_web_sm")
# ents = [ent.text for doc in nlp.pipe(df['text'].astype(str), batch_size=50) for ent in doc.ents]
# ent_df = pd.DataFrame(Counter(ents).most_common(20), columns=['entity','count'])
# st.dataframe(ent_df)

# === added document clustering
st.subheader('Document Clustering (TF–IDF + SVD + KMeans)')
# 1. Cluster into 5 groups
n_clusters = 5                                           # <<< ADDED
kmeans = KMeans(n_clusters=n_clusters, random_state=0)   # <<< ADDED
clusters = kmeans.fit_predict(tfidf)                     # <<< ADDED

# 2. Dimensionality reduction for visualization
svd = TruncatedSVD(n_components=2, random_state=0)       # <<< ADDED
coords_svd = svd.fit_transform(tfidf)                        # <<< ADDED

# 3. Scatter plot of clusters
fig3, ax3 = plt.subplots()                               # <<< ADDED
for label in range(n_clusters):                         # <<< CHANGED: loop per cluster
    idx = clusters == label
    ax3.scatter(
        coords_svd[idx, 0],
        coords_svd[idx, 1],
        alpha=0.5,
        label=f'Cluster {label}'
    )
ax3.set_xlabel('Component 1')                            # <<< ADDED
ax3.set_ylabel('Component 2')                            # <<< ADDED
ax3.set_title('Document Clusters')                       # <<< ADDED
ax3.legend()                                            # <<< CHANGED: add legend
st.pyplot(fig3)                                          # <<< ADDED

# 4. Cluster sizes
cluster_counts = pd.Series(clusters).value_counts().sort_index()  # <<< ADDED
st.write('Cluster counts:', cluster_counts.to_dict())            # <<< ADDED

# === added UMAP visualization
st.subheader('Document Clustering (UMAP)')
umap_model = umap.UMAP(n_components=2, random_state=0)
umap_coords = umap_model.fit_transform(tfidf.toarray())

fig4, ax4 = plt.subplots()
for label in range(n_clusters):                         # <<< CHANGED: loop per cluster
    idx = clusters == label
    ax4.scatter(
        umap_coords[idx, 0],
        umap_coords[idx, 1],
        alpha=0.5,
        label=f'Cluster {label}'
    )
ax4.set_xlabel('UMAP 1')                                # <<< ADDED
ax4.set_ylabel('UMAP 2')                                # <<< ADDED
ax4.set_title('Document Clusters (UMAP)')               # <<< ADDED
ax4.legend()                                            # <<< CHANGED: add legend
st.pyplot(fig4)                                         # <<< ADDED
