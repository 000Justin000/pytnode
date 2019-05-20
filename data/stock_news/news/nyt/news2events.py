import json
import dateutil.parser
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
    headlines = []
    events = []

    nonkeywords = ["Applebee"]
    keywords = ["Apple", "Steve Jobs", "Steve P. Jobs", "Tim Cook", "Tim D. Cook", "iPod", "iPad", "iPhone", "iMac", "Macintosh", "MacBook"]
    
    with open("nyt20071") as json_file: 
        data = json.load(json_file) 

    for article in data["response"]["docs"]:
        headline = getstr(article["headline"], "main")
        if contains(removes(headline, nonkeywords), keywords):
            articles.append(article)
            headlines.append(headline)
            ss = sid.polarity_scores(headline)
            tt = dateutil.parser.isoparse(article["pub_date"]).timestamp()
            events.append((tt, ss["compound"]))
