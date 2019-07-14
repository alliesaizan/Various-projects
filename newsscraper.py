#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun  1 14:46:48 2019

@author: alliesaizan
"""

##############################################
# Package import
from eventregistry import *
import spacy
import textacy
import nltk
nltk.download('vader_lexicon')
from nltk.sentiment.vader import SentimentIntensityAnalyzer as SIA
import os
import re
import pandas as pd
pd.set_option('display.max_columns', 500)
import pickle
from itertools import chain

os.chdir("/Users/alliesaizan/Documents/Python-Tinkering/Pudding")


##############################################
# Helper Functions

def find_sentence_objects(tagged):
    """
    This function finds the direct objects in the article title.
    """
#    tokens  = [i.text for i in tagged]
#    start = [i for (i, w) in enumerate(tokens) if bool(re.search("millennial", w.lower())) == True][0]
    try:
        objs = [i.text for i in tagged.noun_chunks if bool(re.search("dobj", i.root.dep_)) == True]
    except:
        objs = ""
    return(objs)

def findall(sub, lst, overlap = True):
    """
    This function finds the indicies where a sub-list occurs in a larger list.
    I adapted this function from:
    http://paddy3118.blogspot.com/2014/06/indexing-sublist-of-list-way-you-index.html
    """
    sublen = len(sub)
    firstthing = sub[0] if sub else []
    indices, indx = [], -1
    while True:
        try:
            indx = lst.index(firstthing, indx + 1)
        except ValueError:
            break
        if sub == lst[indx : indx + sublen]:
            indices.append(indx)
            if not overlap:
                indx += sublen - 1
    return(indices)

pattern = "(AUX\s)*(ADV\s)*(PART\s)*(VERB\s)+(ADV\s)*(PART\s)*"

def find_verb_phrases(doc):
    """
    This function is designed to pull verb phrases from sentences where 
    millennials are the subject of the sentence. It assumes that the first 
    verb or verb phrase will refer to actions taken by millennials.
    """
    # Obtain the parts of speech tags for each word in the title
    pos_tags = " ".join([i.pos_ for i in doc])
    
    # If the verb phrase pattern matches anywhere in the part of speech tags:
    if re.search(pattern, pos_tags): 
        # Find the matching tags and extract them as a list of tags.
        compiled = re.search(pattern, pos_tags)
        compare_this = compiled.group().split()
        # Compare this list of tags against the full list of tags for all the words in the title.
        # Extract the indicies where the tags occur in the list.
        result = findall(compare_this, pos_tags.split())[0]
        # In the title, pull out the words with matching indicies.
        verbs = " ".join([i.text for i in doc][result:result + len(compare_this)])
    else:
        # If the title does not contain any verb phrases, just extract the first verb in the sentence
        verbs = [i.text for i in doc if i.pos_ == "VERB"]
        if len(verbs) != 0:
            verbs = verbs[0]
        else:
            # If the sentence does not contain any words tagged as verbs, return an empty string
            verbs = ""
    return(verbs)



##############################################
# Instantiate Event Registry API

api_key = "b3b5aa5d-a173-4102-97e6-227c795f7349"

er = EventRegistry(apiKey = api_key)

#q = QueryArticlesIter(
#    keywords = QueryItems.OR(["millennials", "Millennials", "millenial", "Millenial"]),
#    lang = "eng",
#    keywordsLoc="title",
#    dateStart = datetime.date(2015, 6, 16),
#    dateEnd = datetime.date(2015, 10, 11),
##    startSourceRankPercentile = 0,
##    endSourceRankPercentile = 20,
#    dataType = ["news"])

#articles = pd.DataFrame(columns = ["title", "url", "text", "date"])

#for art in q.execQuery(er, sortBy = "date"):    
#    articles = articles.append({ "title": art["title"], "url": art["url"], "text": art["body"], "date": art["date"]}, ignore_index = True)

pickle.dump(articles, open("articles.pkl", "wb"))

# Export sample data
df_sample = articles.sample(n = 100)
df_sample.to_csv("Sample Articles.csv", index = False)


##############################################
# Cleaning and feature creation

articles = pickle.load(open("articles.pkl", "rb"))

# Some text cleaning
articles["title"] = articles["title"].replace("&#\d+|\(|\)", "", regex = True)
articles["title"] = articles.title.apply(lambda x: re.split("\s*(\||;|\.|\s-\s)", str(x))[0])  # split on "|", ";","."

articles["title_lower"] = articles["title"].apply(lambda x: x.lower())

articles = articles[["title", "url", "text", "date", "title_lower", "tagged"]]
articles.drop_duplicates(inplace = True)

nlp = spacy.load("en_core_web_sm")
articles["tagged"] = articles["title"].apply(nlp)

articles["verbs"] = articles.tagged.apply(find_verb_phrases)

articles["objects"] = articles.tagged.apply(find_sentence_objects)
articles["objects"] = articles.objects.replace("^\s+", "", regex= True)

articles["subject"] = articles.tagged.apply(lambda x: [token.text.lower() for token in x if token.dep_ in ["nsubj", "ROOT"]])
articles["mil_subj"] = articles.subject.apply(lambda x: 1 if "millennials" in str(x).lower() else 0)

millennial_articles = articles.loc[articles["mil_subj"] == 1]

# Export objects to CSV
verbs = millennial_articles.verbs.tolist()
verbs = [i.lower() for i in list(chain.from_iterable(verbs))]
verbs = set(verbs)
verbs = list(verbs)

with open("verbs.txt", "w") as f:
    for verb in verbs:
        f.write("%s\n" % verb )
f.close()

objects = millennial_articles.objects.tolist()
objects = [i.lower() for i in list(chain.from_iterable(objects))]
objects = set(objects)
objects = list(objects)

with open("objects.txt", "w") as f:
    for obj in objects:
        f.write("%s\n" % obj )
f.close()

del verb, obj, verbs, objects


##############################################
# Group verbs by noun chunks and vice versa
millennial_articles["objects_len"] = millennial_articles["objects"].apply(lambda x: len(x))
millennial_articles.loc[millennial_articles.objects_len == 0, "objects"] = ""

articles_new = millennial_articles[["title_lower", "verbs", "objects"]].set_index(["title_lower","verbs"])["objects"].apply(pd.Series).stack()
articles_new = articles_new.reset_index()

articles_new = articles_new.drop(labels = "level_2", axis = 1).drop_duplicates()
articles_new.columns = ["title_lower", "verbs", "objects"]
articles_new = articles_new.append(millennial_articles.loc[millennial_articles.objects_len == 0, ["title_lower", "verbs", "objects"]].drop_duplicates())

articles_new["article_id"] = articles_new.index

articles_new["verbs"] = articles_new["verbs"].apply(lambda x: x.lower())
articles_new["objects"] = articles_new["objects"].apply(lambda x: x.lower())

grouped_by_verbs = articles_new.loc[articles_new.objects != ""].groupby('verbs')['objects'].apply(set).reset_index()
grouped_by_verbs["length"] =  grouped_by_verbs["objects"].apply(lambda x: len(x))

grouped_by_objs = articles_new.loc[articles_new["verbs"] != ""].groupby('objects')['verbs'].apply(set).reset_index()
grouped_by_objs["length"] =  grouped_by_objs["verbs"].apply(lambda x: len(x))


grouped_by_verbs.to_csv("Grouped_Objects.csv", index = False)
grouped_by_objs.to_csv("Grouped_Verbs.csv", index = False)


##############################################
# Sentiment analysis
analyzer = SIA()
millennial_articles["polarity"] = millennial_articles["title"].apply(lambda x: analyzer.polarity_scores(x)["compound"])
negative_articles = millennial_articles.loc[millennial_articles["polarity"] < 0]

negative_articles[["title", "polarity"]].to_csv("Negative_Millennial_Articles.csv", index = False)

with open("verbs.txt", "w") as f:
    for verb in negative_verbs:
        f.write("%s\n" % verb )
f.close()




# Find URL domains
publications = pd.read_csv("publications.csv")
websites = publications["domain"].tolist()

domains = [re.split("//(www\.)*", x)[-1] for x in websites]
domains = [x.split(".")[0] for x in domains]

articles["domain"] = articles["url"].apply(lambda x: re.split("//(www\.)*", x)[-1].split(".")[0])
articles["top50_publisher"] = pd.np.where(articles["domain"].isin(domains), 1, 0)

pos_verbs = ["reviv", "using", "search", "driv", "consider", "attract", "balance",\
             "refinanc", "bought", "hoard", "acquire", "purchase", "launch", "embrace",
             "motivat","revamp", "save"]
neg_verbs = ["messed", "delay", "bankrupt", "ignor", "criticize", "destroy", \
         "offend", "recalculate", "scam", "crush", "fall", "rid", "outrag", \
         "betray", "plummet", "divid", "prevent", "bash", "choos", "hurt", \
         "hate", "eliminat", "lack", "wallow", "limit", "discourag", "ruin", \
         "complain", "ban", "doom", "denounc", "refuse", "criticiz", \
         "disgust", "doubt", "derail", "plague", "threaten", "offend", \
         "detach", "delete", "disappear"]

