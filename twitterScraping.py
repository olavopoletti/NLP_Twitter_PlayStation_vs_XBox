# Imports
import os
import time


# The period under analysis - breaking it into two-year periods to produce smaller files
dates = [(f"20{y:02}" + "-06-15", f"20{y + 2:02}" + "-06-14") \
         for y in range(6, 21, 2)] \
        + [("2022-06-15", "2022-08-15")]

# Terms searched
terms = ["xbox", "ps1", "ps2", "ps3", "ps4", "ps5", "Playstation"]

# The parameters to be used by Snscrape - it will use the CLI
scrape = 'snscrape --jsonl --since {} twitter-search "{} until:{}">' \
        + 'tweets_{}_{}_{}.json'

# Scraping for each search term
for t in terms:
    for d in dates:
        os.system(scrape.format(d[0], t, d[1], d[0], d[1], t))
        time.sleep(3600)