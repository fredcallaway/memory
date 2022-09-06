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
	
# stats: stats/exp1 stats/exp2


figs/exp1:
	Rscript exp1.r --out exp1 --models 'optimal,flexible' 

figs/exp1_alt:
	Rscript exp1.r --out exp1_alt --models 'optimal,empirical_old' 


figs/exp2: exp2.r
	Rscript exp2.r --out exp2 --models 'fixed_optimal,flexible' 

figs/exp2_alt: exp2.r
	Rscript exp2.r --out exp2_alt --models 'fixed_optimal,fixed_empirical_old'