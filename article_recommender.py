import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
###### helper functions. Use them when needed #######
def get_title_from_index(index):
	return df[df.index == index]["Title"].values[0]

def get_First_author_from_index(index):
	return df[df.index == index]["First author"].values[0]

def get_index_from_title(Title):
	return df[df.Title == Title]["index"].values[0]
##################################################

##Read CSV File
df = pd.read_csv("article.csv")
## print(df.iloc[14, 2])
##Select Features

features = ['Title','Abstract','Keywords']
## Create a column in DF which combines all selected features
for feature in features:
	df[feature] = df[feature].fillna('')

def combine_features(row):
	try:
		return row['Title'] +" "+row['Abstract']+" "+row["Keywords"]
	except:
		print ("Error:", row)	

df["combined_features"] = df.apply(combine_features,axis=1)

## Create count matrix from this new combined column
cv = CountVectorizer()

count_matrix = cv.fit_transform(df["combined_features"])

## Compute the Cosine Similarity based on the count_matrix
cosine_sim = cosine_similarity(count_matrix) 
## val = input("Enter Title: ") 
val = df.iloc[-1]["Title"]

## Get index of this article from its title
article_index = get_index_from_title(val)

similar_articles =  list(enumerate(cosine_sim[article_index]))

## Get a list of similar articles in descending order of similarity score
sorted_similar_articles = sorted(similar_articles,key=lambda x:x[1],reverse=True)
# print (similar_articles)
# Print titles of first 3 articles
print ("Recomended artcles are:")
i=0
for element in sorted_similar_articles:
		print (get_title_from_index(element[0]) + " By " + get_First_author_from_index(element[0] ))
		i=i+1
		if i>5:
			break
