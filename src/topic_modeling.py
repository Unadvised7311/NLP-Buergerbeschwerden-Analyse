from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import NMF, LatentDirichletAllocation

def run_topic_modeling(data):
    min_df = 5
    max_df = 0.9
    n_topics = 5

    tfidf_vectorizer = TfidfVectorizer(max_df=max_df, min_df=min_df)
    tfidf_matrix = tfidf_vectorizer.fit_transform(data)
    nmf_model = NMF(n_components=n_topics, random_state=42).fit(tfidf_matrix)

    count_vectorizer = CountVectorizer(max_df=max_df, min_df=min_df)
    count_matrix = count_vectorizer.fit_transform(data)
    lda_model = LatentDirichletAllocation(n_components=n_topics, random_state=42).fit(count_matrix)

    return (tfidf_vectorizer, nmf_model), (count_vectorizer, lda_model), count_matrix
