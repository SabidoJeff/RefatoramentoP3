import nltk
from nltk.corpus import conll2000
from nltk.classify import megam
import os
class UnigramChunker(nltk.ChunkParserI):

    def __init__(self, train_sents):
        train_data = [[ (t,c) for w,t,c in nltk.chunk.tree2conlltags(sent)]
            for sent in train_sents]
        self.tagger = nltk.UnigramTagger(train_data)
    def parse(self, sentence):
        pos_tags = [pos for (word, pos) in sentence]
        tagged_pos_tags = self.tagger.tag(pos_tags)
        chunktags = [chunktag for (pos, chunktag) in tagged_pos_tags]
        conlltags = [(word, pos, chunktag) for ((word, pos), chunktag)
                    in zip (sentence, chunktags)]
        return nltk.chunk.conlltags2tree(conlltags)

def tags_since_dt(sentence, i):
    tags = set()
    for word, pos in sentence[:i]:
        if (pos == 'DT'):
            tags = set()
        else:
            tags.add(pos)
    return '+'.join(sorted(tags))

def npchunk_features(sentence, i , history):
    word, pos = sentence[i]
    if(i==0):
        prevword, prevpos = "<START>", "<START>"
    else:
        prevword, prevpos = sentence[i-1]
    if i == len(sentence) - 1:
        nextword, nextpos = "<END>", "<END>"
    else:
        nextword, nextpos = sentence[i+1]
    return {"pos":pos, "word":word, "prevpos":prevpos, "nextpos":nextpos, "prevpos+pos": "%s+%s" % (prevpos, pos),
            "pos+nextpos": "%s+%s" % (pos, nextpos), "tags-since-dt": tags_since_dt(sentence, i)}

class ConsecutiveNPChunkTagger(nltk.TaggerI):

    def __init__(self, train_sents):
        megam_path = os.path.expanduser('/home/jeferson/Documentos/')
        megam.config_megam(megam_path)
        train_set = []
        for tagged_sent in train_sents:
            untagged_sent = nltk.tag.untag(tagged_sent)
            history = []
            for i, (word, tag) in enumerate(tagged_sent):
                featureset = npchunk_features(untagged_sent, i, history)
                train_set.append( (featureset, tag) )
                history.append(tag)
        self.classifier = nltk.MaxentClassifier.train(train_set, algorithm= 'megam', trace=0)

    def tag(self, sentence):
        history = []
        for i, word in enumerate(sentence):
            featureset = npchunk_features(sentence, i, history)
            tag = self.classifier.classify(featureset)
            history.append(tag)
        return zip(sentence, history)

class ConsecutiveNPChunker(nltk.ChunkParserI):
    def __init__(self, train_sents):
        tagged_sents = [[((w,t),c) for (w,t,c) in
                         nltk.chunk.tree2conlltags(sent)]
                        for sent in train_sents]
        self.tagger = ConsecutiveNPChunkTagger(tagged_sents)

    def parse(self, sentence):
        tagged_sents = self.tagger.tag(sentence)
        conlltags = [(w,t,c) for ((w,t),c) in tagged_sents]
        return nltk.chunk.conlltags2tree(conlltags)

#This function receives a raw text and return it splitted in sentences
def sentences(text):
    return nltk.sent_tokenize(text)

#This function receives a list of sentences and split all sentences in a list of words
def tokenization(sentences):
    return [nltk.word_tokenize(sent) for sent in sentences]

#This function recieves a list of list of words (sentences) and pos_tag all words
def tagging(tokenized):
    return [nltk.pos_tag(sent) for sent in tokenized]

#This function receives a list of pos_tagged words (a sentence)
#and create a chunked tagged list of words, using a noun as a split condition 
def noun_chunking(tagged):
    grammar = r"""
    NP: {<DT|PP\$>?<JJ>*<NN>}
    {<NNP>+}
    """
    cp = nltk.RegexpParser(grammar)
    result = cp.parse(tagged)
    return result

def main():
    file1 = open("S0950584908001390.txt", 'r')
    text = file1.read()
    file1.close()
    text = sentences(text)
    text = tokenization(text)
    text = tagging(text)
    ##tree = noun_chunking(text[0])
    train_sents = conll2000.chunked_sents('train.txt', chunk_types=['NP'])
    test_sents = conll2000.chunked_sents('test.txt', chunk_types=['NP'])
    #unigram_chunker = UnigramChunker(train_sents)
    #tree = unigram_chunker.parse(text[0])
    #tree.draw()
    chunker = ConsecutiveNPChunker(train_sents)
    print(chunker.evaluate(test_sents))

if __name__=="__main__":
    main() 