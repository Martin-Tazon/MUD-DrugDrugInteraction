#! /usr/bin/python3

import sys
import re
from os import listdir

from xml.dom.minidom import parse
from nltk.tokenize import word_tokenize

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

def isCamel(s):
    return (s != s.lower() and s != s.upper() and "_" not in s)

def isFirstCap(s):
    return s[0].isupper() and s[1:].islower()

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
        
        ### Current token features
        tokenFeatures.append("form="+t)
        tokenFeatures.append("suf3="+t[-3:])
        tokenFeatures.append("suf6="+t[-6:])
        
        # Types of cases
        tokenFeatures.append("lowercase="+str(t.islower()))
        tokenFeatures.append("uppercase="+str(t.isupper()))
        tokenFeatures.append("camelcase="+str(isCamel(t)))
        tokenFeatures.append("firstuppercase="+str(isFirstCap(t)))

        ## Change to lowercase
        t = t.lower()
        
        # Has prefix or suffix
        tokenFeatures.append("hasPrefix="+str(any(t.startswith(p) for p in prefixes)))
        tokenFeatures.append("hasSuffix="+str(any(t.endswith(s) for s in suffixes)))
        
        # Get length
        tokenFeatures.append("len="+str(len(t)))
        # Is the token large
        tokenFeatures.append("longToken="+str(len(t)>8))
        
        ## Check lookup files
        #tokenFeatures.append("isDrug="+str("drug" == lookupDrugs[t.lower()] if t.lower() in lookupDrugs.keys() else "F"))
        #tokenFeatures.append("isDrugN="+str("drug_n" == lookupDrugs[t.lower()] if t.lower() in lookupDrugs.keys() else "F"))
        #tokenFeatures.append("isBrand="+str("brand" == lookupDrugs[t.lower()] if t.lower() in lookupDrugs.keys() else "F"))
        #tokenFeatures.append("isGroup="+str("group" == lookupDrugs[t.lower()] if t.lower() in lookupDrugs.keys() else "F"))

        # Check lookup files V2
        for kind in lookupDrugs.keys():
            known = lookupDrugs[kind]
            inDBank = t in known
            if inDBank:
                tokenFeatures.append(f"inDB{kind}=True")
            else:
                asPart = any([ t in k.split() for k in known ])
                if asPart:
                    tokenFeatures.append(f"inDB{kind}=Partial")
                else:
                    tokenFeatures.append(f"inDB{kind}=False")
        
        # Numeric characters
        tokenFeatures.append("hasNumbers="+str(any(c.isdigit() for c in t)))
        tokenFeatures.append("isNumbers="+str(t.isdigit()))
        
        ### Containes dashes or parantheses    # The following gives the same info.
        #tokenFeatures.append("hasDashes="+str('-' in t))
        #tokenFeatures.append("hasOpenPar="+str('(' in t))
        #tokenFeatures.append("hasClosePar="+str(')' in t))
        tokenFeatures.append("hasSymbols="+str(any(c in symbols for c in t)))
        
        ## Number of dashes
        #tokenFeatures.append("numDash="+str(t.lower().count('-')))
        #tokenFeatures.append("numOpenPar="+str(t.lower().count('(')))
        #tokenFeatures.append("numClosePar="+str(t.lower().count(')')))
        
        # Contains non-alphanumeric
        tokenFeatures.append("isAlphaNum="+str(t.isalnum()))
        
        ## Number of x, y and z
        #tokenFeatures.append("numX="+str(t.lower().count('x')))
        #tokenFeatures.append("numY ="+str(t.lower().count('y')))
        #tokenFeatures.append("numZ="+str(t.lower().count('z')))
        
        # Ratio of capital letters
        # tokenFeatures.append("ratioCaps="+str(capitalRatio(t) > 0.5))

        # Has a term useful to identify groups --> Reduces drug_n
        #tokenFeatures.append("hasGroupTerm="+str(t in termsForGroups))
        
        ####################
        ### Previous token features
        
        if k>0 :
          tPrev = tokens[k-1][0].lower()
          tokenFeatures.append("formPrev="+tPrev)
          tokenFeatures.append("suf3Prev="+tPrev[-3:])
          tokenFeatures.append("suf6Prev="+tPrev[-6:])
          tokenFeatures.append("lenPrev="+str(len(tPrev)))
          tokenFeatures.append("hasNumbersPrev="+str(any(c.isdigit() for c in tPrev)))
          #tokenFeatures.append("hasPrefixPrev="+str(any(tPrev.startswith(p) for p in prefixes)))
          #tokenFeatures.append("hasSuffixPrev="+str(any(tPrev.endswith(s) for s in suffixes)))
          #tokenFeatures.append("hasSymbolsPrev="+str(any(c in symbols for c in tPrev)))
        else :
          tokenFeatures.append("BoS")
        
        if k>1 :
          tPrev2 = tokens[k-2][0].lower()
          tokenFeatures.append("formPrev2="+tPrev2)
          tokenFeatures.append("suf3Prev2="+tPrev2[-3:])
          tokenFeatures.append("lenPrev2="+str(len(tPrev2)))
          tokenFeatures.append("hasNumbersPrev2="+str(any(c.isdigit() for c in tPrev2)))
        else:
          pass # add something?
        
        if k>2 :
          tPrev3 = tokens[k-3][0].lower()
          tokenFeatures.append("formPrev3="+tPrev3)
          tokenFeatures.append("suf3Prev3="+tPrev3[-3:])
          tokenFeatures.append("lenPrev3="+str(len(tPrev3)))
          tokenFeatures.append("hasNumbersPrev3="+str(any(c.isdigit() for c in tPrev3)))
        else:
          pass # add something?
        
        
        ### Next token features
        
        if k<len(tokens)-1 :
          tNext = tokens[k+1][0].lower()
          tokenFeatures.append("formNext="+tNext)
          tokenFeatures.append("suf3Next="+tNext[-3:])
          tokenFeatures.append("lenNext="+str(len(tNext)))
          tokenFeatures.append("hasNumbersNext="+str(any(c.isdigit() for c in tNext)))
          #tokenFeatures.append("hasGroupTermNext="+str(tNext in termsForGroups))
          #tokenFeatures.append("hasPrefixNext="+str(any(tNext.startswith(p) for p in prefixes)))
          #tokenFeatures.append("hasSuffixNext="+str(any(tNext.endswith(s) for s in suffixes)))
          #tokenFeatures.append("hasSymbolsNext="+str(any(c in symbols for c in tNext)))
        else:
          tokenFeatures.append("EoS")
        
        if k<len(tokens)-2 :
          tNext2 = tokens[k+2][0].lower()
          tokenFeatures.append("formNext2="+tNext2)
          tokenFeatures.append("suf3Next2="+tNext2[-3:])
          tokenFeatures.append("lenNext2="+str(len(tNext2)))
          tokenFeatures.append("hasNumbersNext2="+str(any(c.isdigit() for c in tNext2)))
          #tokenFeatures.append("hasPrefixNext2="+str(any(tNext2.startswith(p) for p in prefixes)))
          #tokenFeatures.append("hasSuffixNext2="+str(any(tNext2.endswith(s) for s in suffixes)))
          #tokenFeatures.append("hasSymbolsNext2="+str(any(c in symbols for c in tNext2)))
        else:
          pass
        
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

prefixes = ['acetyl', 'amino', 'anti', 'azo', 'bromo', 'cyclo', 'deoxy', 'di', 'dihydro', 'erythro', 'fluoro', 'hydroxy', 'iso', 'lipo', 'meta', 'methyl', 'neo', 'ortho', 'para', 'phenyl', 'phospho', 'pro', 'pyrro', 'sulfo', 'thio', 'trans', 'tri']
suffixes = ['amine', 'azole', 'cillin', 'cycline', 'dazole', 'dine', 'dronate', 'fenac', 'fil', 'floxacin', 'gliptin', 'ine', 'lamide', 'mab', 'nib', 'ol', 'oprazole', 'oxacin', 'parin', 'phylline', 'prazole', 'ridone', 'sartan', 'setron', 'statin', 'tidine', 'triptan', 'vastatin', 'vir', 'zepam', 'zide', 'zole']

termsForGroups = ['drugs', 'medicines', 'agents', 'supplements', 'medications', 'products', 'preparation', 'agonists', 'adjuvants', 'antagonists', 'blockers', 'inhibitors']
symbols = ["[","]","(",")","{","}","-","_"]


lookupDrugs = {}
with open(datadir+"/../../resources/DrugBank.txt", encoding="utf-8") as f:
    for d in f.readlines():
        (t, c) = d.strip().lower().split("|")
        if c not in lookupDrugs:
            lookupDrugs[c] = [t]
        else:
            lookupDrugs[c].append(t)


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
