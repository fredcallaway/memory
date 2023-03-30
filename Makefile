PAPER = ~/Papers/meta-memory
RUN = sep11

sync: # sends results to the paper directory, kind of abusing make here...
	rsync --delete-after -av analysis/stats/ $(PAPER)/stats/
	rsync -av model/results/$(RUN)/exp1/tex/ $(PAPER)/stats/exp1/
	rsync -av model/results/$(RUN)/exp2/tex/ $(PAPER)/stats/exp2/
	rsync -av model/results/$(RUN)/old_exp1/tex/ $(PAPER)/stats/old_exp1/
	rsync -av model/results/$(RUN)/old_exp2/tex/ $(PAPER)/stats/old_exp2/

	rsync --delete-after --exclude *.png --exclude tmp* -av analysis/figs/$(RUN)/ $(PAPER)/figs/

	rsync -av --delete-after model/figs/ $(PAPER)/model-diagram/model-figs/

all:
	cd data && make processed
	cd model && make results/$(RUN)
	cd analysis && make figs/$(RUN)
	cd analysis && make stats
