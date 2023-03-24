#! /usr/bin/python3

import sys
import re
from os import listdir

from xml.dom.minidom import parse
from nltk.tokenize import word_tokenize

prefixes = ['acetyl', 'amino', 'anti', 'azo', 'bromo', 'cyclo', 'deoxy', 'di', 'dihydro', 'erythro', 'fluoro', 'hydroxy', 'iso', 'lipo', 'meta', 'methyl', 'neo', 'ortho', 'para', 'phenyl', 'phospho', 'pro', 'pyrro', 'sulfo', 'thio', 'trans', 'tri']
suffixes = ['amine', 'azole', 'cillin', 'cycline', 'dazole', 'dine', 'dronate', 'fenac', 'fil', 'floxacin', 'gliptin', 'ine', 'lamide', 'mab', 'nib', 'ol', 'oprazole', 'oxacin', 'parin', 'phylline', 'prazole', 'ridone', 'sartan', 'setron', 'statin', 'tidine', 'triptan', 'vastatin', 'vir', 'zepam', 'zide', 'zole']


lookupDrugs = {}
with open("../lab1/DDI/resources/HSDB.txt", encoding="utf-8") as f:
    for d in f.readlines():
        lookupDrugs[d.strip().lower()] = "drug"
with open("../lab1/DDI/resources/DrugBank.txt", encoding="utf-8") as f:
    for d in f.readlines():
        (t, c) = d.strip().lower().split("|")
        lookupDrugs[t] = c


## --------- tokenize sentence ----------- 
## -- Tokenize sentence, returning tokens and span offsets

def tokenize(txt):
    offset = 0
    tks = []
    ## word_tokenize splits words, taking into account punctuations, numbers, etc.
    for t in word_tokenize(txt):
        ## keep track of the position where each token should appear, and
        ## store that information with the token
        offset = txt.find(t, offset)
        tks.append((t, offset, offset+len(t)-1))
        offset += len(t)

    ## tks is a list of triples (word,start,end)
    return tks


## --------- get tag ----------- 
##  Find out whether given token is marked as part of an entity in the XML

def get_tag(token, spans) :
   (form,start,end) = token
   for (spanS,spanE,spanT) in spans :
      if start==spanS and end<=spanE : return "B-"+spanT
      elif start>=spanS and end<=spanE : return "I-"+spanT

   return "O"

def iscamel(s):
   return (s != s.lower() and s != s.upper() and "_" not in s)

def capitalRatio(s):
    capitalLetters = sum(1 for c in s if c.isupper())
    totalLetters = len(s)
    if totalLetters == 0:
        return 0
    else:
        return capitalLetters / totalLetters
 
## --------- Feature extractor ----------- 
## -- Extract features for each token in given sentence

def extract_features(tokens) :

   # for each token, generate list of features and add it to the result
   result = []
   for k in range(0,len(tokens)):
      tokenFeatures = []
      t = tokens[k][0]

      # New token features
      # Get length
      tokenFeatures.append("len="+str(len(t)))

      # Types of cases
      tokenFeatures.append("lowercase="+str(t.islower()))
      tokenFeatures.append("uppercase="+str(t.isupper()))
      # tokenFeatures.append("titlecase="+str(t.istitle()))
      tokenFeatures.append("camelcase="+str(iscamel(t)))
      tokenFeatures.append("hasPrefix="+str(t in prefixes))
      tokenFeatures.append("hasSuffix="+str(t in suffixes))

      # Check lookup files
      tokenFeatures.append("isDrug="+str("drug" == lookupDrugs[t.lower()] if t.lower() in lookupDrugs.keys() else "F"))
      tokenFeatures.append("isDrugN="+str("drug_n" == lookupDrugs[t.lower()] if t.lower() in lookupDrugs.keys() else "F"))
      tokenFeatures.append("isBrand="+str("brand" == lookupDrugs[t.lower()] if t.lower() in lookupDrugs.keys() else "F"))
      tokenFeatures.append("isGroup="+str("group" == lookupDrugs[t.lower()] if t.lower() in lookupDrugs.keys() else "F"))

      # Numeric characters
      tokenFeatures.append("hasNumbers="+str(any(c.isdigit() for c in t)))
      
      # Containes dashes or parantheses
      tokenFeatures.append("hasDashes="+str('-' in t))
      tokenFeatures.append("hasOpenPar="+str('(' in t))
      tokenFeatures.append("hasClosePar="+str(')' in t))
      
      # Number of dashes
      tokenFeatures.append("numDash="+str(t.lower().count('-')))
      tokenFeatures.append("numOpenPar="+str(t.lower().count('(')))
      tokenFeatures.append("numClosePar="+str(t.lower().count(')')))
      
      # Contains non-alphanumeric
      tokenFeatures.append("isAlphaNum="+str(t.isalnum()))

      # Number of x, y and z
      tokenFeatures.append("numX="+str(t.lower().count('x')))
      tokenFeatures.append("numY ="+str(t.lower().count('y')))
      tokenFeatures.append("numZ="+str(t.lower().count('z')))

      # Ratio of capital letters
      # tokenFeatures.append("ratioCaps="+str(capitalRatio(t) > 0.5))

      
      if k<len(tokens)-1 :
         tokenFeatures.append("nextHasNumbers="+str(any(c.isdigit() for c in tokens[k+1][0])))
         


      ####################

      tokenFeatures.append("form="+t)
      tokenFeatures.append("suf3="+t[-3:])

      if k>0 :
         tPrev = tokens[k-1][0]
         tokenFeatures.append("formPrev="+tPrev)
         tokenFeatures.append("suf3Prev="+tPrev[-3:])
      else :
         tokenFeatures.append("BoS")

      if k<len(tokens)-1 :
         tNext = tokens[k+1][0]
         tokenFeatures.append("formNext="+tNext)
         tokenFeatures.append("suf3Next="+tNext[-3:])
      else:
         tokenFeatures.append("EoS")

      # Add new features
      #tokenFeatures.append(f"len={len(t)}")
      # Is it in DrugBank?
      # Capitalization
      # Following token
      # Following is numeric (or numeric+text)
    
      result.append(tokenFeatures)
    
   return result


## --------- MAIN PROGRAM ----------- 
## --
## -- Usage:  baseline-NER.py target-dir
## --
## -- Extracts Drug NE from all XML files in target-dir, and writes
## -- them in the output format requested by the evalution programs.
## --


# directory with files to process
datadir = sys.argv[1]

# process each file in directory
for f in listdir(datadir) :
   
   # parse XML file, obtaining a DOM tree
   tree = parse(datadir+"/"+f)
   
   # process each sentence in the file
   sentences = tree.getElementsByTagName("sentence")
   for s in sentences :
      sid = s.attributes["id"].value   # get sentence id
      spans = []
      stext = s.attributes["text"].value   # get sentence text
      entities = s.getElementsByTagName("entity")
      for e in entities :
         # for discontinuous entities, we only get the first span
         # (will not work, but there are few of them)
         (start,end) = e.attributes["charOffset"].value.split(";")[0].split("-")
         typ =  e.attributes["type"].value
         spans.append((int(start),int(end),typ))
         

      # convert the sentence to a list of tokens
      tokens = tokenize(stext)
      # extract sentence features
      features = extract_features(tokens)

      # print features in format expected by crfsuite trainer
      for i in range (0,len(tokens)) :
         # see if the token is part of an entity
         tag = get_tag(tokens[i], spans) 
         print (sid, tokens[i][0], tokens[i][1], tokens[i][2], tag, "\t".join(features[i]), sep='\t')

      # blank line to separate sentences
      print()
