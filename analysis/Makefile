sync:
	rsync --delete-after -av stats/ ~/Papers/meta-memory/stats/
	rsync --delete-after --exclude *.png --exclude tmp* -av figs/ ~/Papers/meta-memory/figs/
	rsync --delete-after --exclude *.png --exclude tmp* -av figs/ ~/Papers/meta-memory/figs/

../data/processed/exp1: exp1_preprocess.r
	Rscript exp1_preprocess.r

../data/processed/exp2: exp2_preprocess.r
	Rscript exp2_preprocess.r

stats/exp1: exp1_stats.r ../data/processed/exp1
	Rscript exp1_stats.r

stats/exp2: ../data/processed/exp2
	Rscript exp2_stats.r
	
stats: stats/exp1 stats/exp2
