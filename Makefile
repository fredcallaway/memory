PAPER = ~/Papers/meta-memory
RUN = sep7

all:
	cd data && make processed
	cd model && make results/sep7
	cd analysis && make figs/sep7
	cd analysis && make stats

sync: # sends results to the paper directory, kind of abusing make here...
	rsync --delete-after -av analysis/stats/ $(PAPER)/stats/
	rsync -av model/results/$(RUN)/exp1/tex/ $(PAPER)/stats/exp1/
	rsync -av model/results/$(RUN)/exp2/tex/ $(PAPER)/stats/exp2/

	rsync --delete-after --exclude *.png --exclude tmp* -av analysis/figs/$(RUN) $(PAPER)/figs/
	rsync --delete-after --exclude *.png --exclude tmp* -av analysis/figs/$(RUN) $(PAPER)/figs/
