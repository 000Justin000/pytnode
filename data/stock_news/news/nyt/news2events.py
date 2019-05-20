import json
import dateutil.parser
import numpy as np
from nltk.sentiment.vader import SentimentIntensityAnalyzer

def getstr(dic, key):
    message = ""
    if type(dic) is dict:
        result = dic.get(key, "")
        if result is not None:
            message = result
    return message

def removes(article, keywords):
    for keyword in keywords:
        article = article.replace(keyword, "")
    return article

def contains(article, keywords):
    contain = False
    for keyword in keywords:
        if keyword in article:
            contain = True
    return contain


if __name__ == "__main__":

    sid = SentimentIntensityAnalyzer()
    articles = []
    events = []
    headlines = set()

    nonkeywords = ["Applebee"]
    keywords = ["Apple", "Steve Jobs", "Steve P. Jobs", "Tim Cook", "Tim D. Cook", "iPod", "iPad", "iPhone", "iMac", "Macintosh", "MacBook"]
    
    #------------------------------------------
    for y in range(2006,2019):
        #--------------------------------------
        for m in range(1,13):
            #----------------------------------
            filename = "nyt" + str(y) + str(m)
            #----------------------------------
            with open(filename) as json_file: 
                data = json.load(json_file) 
            #----------------------------------
            for article in data["response"]["docs"]:
                headline = getstr(article["headline"], "main")
                if (headline not in headlines) and contains(removes(headline, nonkeywords), keywords):
                    articles.append(article)
                    ss = sid.polarity_scores(headline)
                    tt = dateutil.parser.isoparse(article["pub_date"]).timestamp()
                    events.append(("{:10d}".format(int(tt)), "{:+10.4f}".format(ss["compound"]), headline))
                    headlines.add(headline)
    #------------------------------------------

    np.savetxt("events", sorted(events), fmt=["%s", "%s", "%s"], delimiter='\t')
