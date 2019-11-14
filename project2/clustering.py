#import os
import collections
import sys
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

def adjMat(root, di, level):
    if root.is_leaf():
        if root.id not in di:
            di[root.id] = set()    
        di[root.id].add(level)
        return di
    if root.left is not None:
        di.update(adjMat(root.left, di, level+1))
    if root.right is not None:
        di.update(adjMat(root.right, di, level+1))
    for i in di:
        if (level+1) in di[i]:
            di[i].add(level)
    return di
  

def main():
    #variables
    simMat=[]
    termMap = {}
    docMap = {}
    topicmap = {}
    #open corpus file
    f=open("/homes/cs473/project2/reut2-subset.sgm","r")
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
            for t in topics:
                if t not in topicmap:
                    topicmap[t] = set()
                topicmap[t].add(d)
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
    #print(n)
    sys.setrecursionlimit(10**6) 
    X = ssd.squareform(simMat)
    Z = linkage(X, 'single')
    singleRoot, singleList = to_tree(Z, rd=True)
    singleAdj = {}
    #os.system("date")
    singleAdj = adjMat(singleRoot, singleAdj, 1)
    changeKey(singleAdj, idIndexMap, cnt)
    #print("analysis")
    #singleScores=findScores(topicmap, singleAdj)
    #print(avg(singleScores))
    #dumpScores(singleScores, "singleAnalysis.txt")

    P = linkage(X, 'complete')
    completeRoot, completeList = to_tree(P, rd=True)
    completeAdj = {}
    completeAdj = adjMat(completeRoot, completeAdj, 1)
    changeKey(completeAdj, idIndexMap, cnt)
    #completeScores=findScores(topicmap, completeAdj)
    #print(avg(completeScores))
    #dumpScores(completeScores, "completeAnalysis.txt")

    writeSortToFile(singleAdj, "single.txt")
    writeSortToFile(completeAdj, "complete.txt")

def avg(di):
    sm=0
    count=0
    for i in di:
        sm+=di[i]
        count+=1
    return sm/count
def dumpScores(di, file):
    f=open(file,"w+")
    f.write("Topics	Score\n")
    dis = collections.OrderedDict(di)
    for i in dis:
        f.write("{t}	{s}\n".format(t=i, s=dis[i]))    
    f.close()

def findScores(topicmap, clusters):
    topicScores={}
    for t in topicmap:
        topicClusts=set()
        for d in topicmap[t]:
            clust=clusters[d.id]
            for c in clust:
                topicClusts.add(c)
        topicScores[t]=len(topicClusts)/len(topicmap[t])
    return topicScores

def changeKey(di, m, r):
    for x in range(r):
        if x in di:
            di[m[x]] = di.pop(x)


def writeSortToFile(di, file):
    f=open(file,"w+")
    f.write("NEWID	clustersID\n")
    for i in sorted(di):
        f.write("{i}	".format(i=i))
        for c in di[i]:
            f.write("{c} ".format(c=c))
        f.write("\n")
        

    f.close()

if __name__ == '__main__':
    main()

