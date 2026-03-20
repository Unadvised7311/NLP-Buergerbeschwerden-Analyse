import spacy
import re

# Laden des Sprachmodells laut Konzept [cite: 9, 15]
nlp = spacy.load("en_core_web_sm", disable=['parser', 'ner'])

def preprocess_pipeline(texts):
    processed_texts = []
    # Domänenspezifische Stoppwörter laut Konzept [cite: 8]
    custom_stopwords = {'dear', 'complaint', 'bank', 'service', 'please', 'xxxx', 'xxxxxxxx'}

    for doc in nlp.pipe(texts, batch_size=500):
        tokens = []
        for token in doc:
            # Entfernung von URLs, Sonderzeichen und Zahlen
            t_low = token.lemma_.lower()
            if (not token.is_stop and not token.is_punct and not token.like_num
                and not token.like_url and len(t_low) > 2
                and t_low not in custom_stopwords):

                # Reinigung von verbliebenen Sonderzeichen
                clean_t = re.sub(r'[^a-zA-Z]', '', t_low)
                if clean_t and not re.fullmatch(r'x+', clean_t):
                    tokens.append(clean_t)

        processed_texts.append(" ".join(tokens))
    return processed_texts
