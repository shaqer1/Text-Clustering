from bs4 import BeautifulSoup
from nltk.stem import PorterStemmer
from nltk.tokenize import sent_tokenize, word_tokenize, RegexpTokenizer
from nltk import FreqDist
from stop_words import get_stop_words
import nltk
nltk.download('stopwords')
nltk.download('punkt')
from nltk.corpus import stopwords
import math 

class Doc:
    def __init__(self, title, body, ID, topics, termFreqMap):
        self.title = title
        self.termFreqMap = termFreqMap
        self.body = body
        self.topics = topics

class Term:
    def __init__(self, t):
        self.t = t
        self.docIDs=[]
        self.docFreq=0
        self.TF={}
        self.IDF=0       

def main():
    #variables
    simMat=[]
    termMap = {}
    docMap = {}
    #open corpus file
    f=open("./Code/reut2-subset.sgm","r")
    text=f.read()
    f.close()
    #pare with soup
    soup = BeautifulSoup(text, "html.parser")
    # find all articles
    docs = soup.findAll("reuters")
    # stop word and stemming processes
    stop_words = set(stopwords.words('english'))
    stop_words.union(set(get_stop_words('en')))
    ps = PorterStemmer()
    #find num of docs
    n = len(docs)
    for i in docs:
        if i.find('body') != None:
            title=i.find('title').text
            id=i.attrs['newid']
            body = i.find('body').text
            topics=[]
            if i.attrs.get('topics',"NO")!="NO":
                topics=i.find('topics').findAll('d')
            termFreq={}
            d = Doc(title,body,id,topics,termFreq)
            #add to docMap
            docMap[id]=d
            #stem and stop word removal
            words=[]
            word_tokens = RegexpTokenizer(r'\w+').tokenize(body)
            words = [w for w in word_tokens if not w in stop_words] 
            words = FreqDist(words)
            d.termFreqMap=words
            #append to term map
            for w in words:
                if w not in termMap:
                    termMap[w]=Term(w)
                #update counts
                termMap[w].docIDs.append(id)
                termMap[w].docFreq+=1
                #update TFIDF
                termMap[w].IDF=math.log(n/termMap[w].docFreq)
                if d not in termMap[w].TF:
                    termMap[w].TF[d]=(math.log(words[w])+1)
                else:
                    print("d already in termMap[w].TF")
            #compute cosine similarity in similarity matrix
            #print(words)

if __name__ == '__main__':
    main()

