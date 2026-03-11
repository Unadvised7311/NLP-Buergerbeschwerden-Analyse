import spacy
import re

nlp = spacy.load("en_core_web_sm", disable=['parser', 'ner'])

def preprocess_pipeline(texts):
    processed_texts = []
    custom_stopwords = {'dear', 'complaint', 'bank', 'service', 'please'}

    for doc in nlp.pipe(texts, batch_size=500):
        tokens = [token.lemma_.lower() for token in doc
                  if not token.is_stop
                  and not token.is_punct
                  and not token.like_num
                  and len(token.text) > 2
                  and token.text.lower() not in custom_stopwords]

        clean_tokens = [re.sub(r'[^a-zA-Z]', '', t) for t in tokens if re.sub(r'[^a-zA-Z]', '', t)]
        processed_texts.append(" ".join(clean_tokens))

    return processed_texts
