import sys
import numpy as np
import pandas as pd
import argparse
import unicodedata
import os

parser = argparse.ArgumentParser()
parser.add_argument('-r', action='store', required=True)
parser.add_argument('-i', action='store', required=True)
args = parser.parse_args()

rspct_df = pd.read_csv(args.r, sep='\t')
info_df = pd.read_csv(args.i)

info_df = info_df[info_df.in_data].reset_index()

i = 0
a = [x for x in range(1, 10)]
for i in range(len(a)):
	a[i] = a[i] * 10

print("Generating files now. This might take a lot of time.")

if not os.path.exists("text_files"):
	os.makedirs("text_files")
for index, rows in rspct_df.iterrows():
	i += 1
	directory = "text_files/"+rows["subreddit"]
	if not os.path.exists(directory):
		os.makedirs(directory)
	if not os.path.exists(directory+"/"+rows["id"]+".txt"):
		with open(directory+"/"+rows["id"]+".txt", "w", encoding="utf-8") as outfile:
			text = rows["selftext"]
			title = rows["title"]
			text2 = text.replace("<lb>", "\n")
			text = text2.replace("<tab>", "\t")
			text = text.replace("& amp;", "&")
			title2 = title.replace("<lb>", "\n")
			title = title2.replace("<tab>", "\t")
			title = title.replace("& amp;", "&")
			outfile.write(title + "\n" + text)
			if i%10000==0 or (i/len(rspct_df.index) in a):
				print(i, i*100/len(rspct_df.index), sep="\t")
