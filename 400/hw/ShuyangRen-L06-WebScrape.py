#!/usr/bin/env python
# coding: utf-8

# This is my submission for L06. I have taken a URL from python.org for the excercise.


# Import the nessessary packages
import requests

response = requests.get("https://wiki.python.org/moin/BeginnersGuide") # Request the page with the URL
content = response.content # What we want is the content

soup = BeautifulSoup(content, "lxml")
# Using BeautifulSoup's prettify() we will get a more readable version of the webpage

# Here we try to list out all the links on this page
# First we use find_all to look inside "a" tags for https tags (aka web links)
all_a_http = soup.find_all("a", "http")   

# Then we use this iterative object to print out all the links and give us a final count.
count = 0
for x in all_a_http:
    print(x.attrs['href'])
    count +=1
# Print final count
print(count)