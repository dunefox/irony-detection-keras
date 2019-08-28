OUT=reddit_shuf_ups.tsv
REDDIT=reddit.tsv
REDDITTEMP=reddit_upsample.tmp
REDDITFINAL=reddit_upsample_shuffle.tsv
SEMEVAL=semeval.tsv
DATA=./data

remove:
	-mkdir $(DATA)
	-rm -f $(DATA)/$(REDDITTEMP) $(DATA)/$(REDDITFINAL)

create: remove
	python misc.py
	shuf $(DATA)/$(REDDITTEMP) > $(DATA)/$(REDDITFINAL)

all: create
	python ./main.py
  # poetry run python misc.py

