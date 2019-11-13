from bs4 import BeautifulSoup
from nltk.stem import PorterStemmer
from nltk.tokenize import sent_tokenize, word_tokenize, RegexpTokenizer
from nltk import FreqDist
from stop_words import get_stop_words
import nltk
nltk.download('stopwords')
nltk.download('punkt')
from nltk.corpus import stopwords
from scipy.cluster.hierarchy import dendrogram, linkage, to_tree
from matplotlib import pyplot as plt
import scipy.spatial.distance as ssd
import math 

class Doc:
    def __init__(self, title, body, ID, topics, termFreqMap):
        self.title = title
        self.termFreqMap = termFreqMap
        self.body = body
        self.id=ID
        self.topics = topics

class Term:
    def __init__(self, t):
        self.t = t
        self.docIDs=[]
        self.docFreq=0
        self.TF={}
        self.IDF=0       
def sumSquare(di):
    sum=0
    for t in di:
        sum+=di[t]*di[t]
    return math.sqrt(sum)

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
    #initialize sim mat
    #simMat = [ [ 0 for i in range(n) ] for j in range(n) ]
    #process each doc/term
    cnt = 0
    idIndexMap={}
    for i in docs:
        if i.find('body') != None:
            title=i.find('title').text
            id=i.attrs['newid']
            #update index in simMat for this doc
            idIndexMap[cnt]=id
            cnt+=1
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
            d.termFreqMap = {x : math.log(y)+1 for x, y in d.termFreqMap.items()}
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
            for i in simMat:
                i.append(0)
            arr=[ 0 for i in range(cnt) ]
            simMat.append(arr)
            #calculate sim
            k=len(simMat)-1
            for j in range(len(simMat)):
                #calculate d1 and d2 sim
                d1Terms=docMap[idIndexMap[k]].termFreqMap
                d2Terms=docMap[idIndexMap[j]].termFreqMap
                simSum=0
                weight=sumSquare(d1Terms)*sumSquare(d2Terms)
                for t in d1Terms:
                    if t in d2Terms:
                        simSum+=(d1Terms[t]*d2Terms[t])
                simMat[k][j]=simSum/weight
                simMat[j][k]=simSum/weight
                if j==k:
                    simMat[k][j]=0
            #print(words)
    print(n)
    X = ssd.squareform(simMat)
    print(len(X))
    Z = linkage(X, 'single')
    Q = to_tree(Z)
    print(Z[0])
    print(Z[1])
    print(Z[2])
    print(Z)
    print(Q[0])
    print(len(Z))
    P = linkage(X, 'complete')
    print(len(P))
    print(cnt)


if __name__ == '__main__':
    main()

