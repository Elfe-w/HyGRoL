corpus = ['parent','child','route','value']
corpus_str = ' '.join(c.replace("\n", "") for c in corpus)
with open('../data/emb/bcb_corpus.txt', 'a', encoding='utf-8') as f:
    f.write(corpus_str)
    f.write('\n')