from nltk.tokenize import WordPunctTokenizer
from gensim.models import Word2Vec


def train_w2v():
    sentences = [['this', 'is', 'the', 'first', 'sentence', 'for', 'word2vec'],
                ['this', 'is', 'the', 'second', 'sentence'],
                ['yet', 'another', 'sentence'],
                ['one', 'more', 'sentence'],
                ['and', 'the', 'final', 'sentence']]

    model = Word2Vec(sentences, min_count=1)

    print(model.wv.get_vector('sentence'))
    print(model.wv.most_similar('first'))


if __name__ == '__main__':
    train_w2v()
