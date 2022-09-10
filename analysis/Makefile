all: stats/ figs/

../data/processed/exp1: exp1_preprocess.r preprocess_common.jl
	Rscript exp1_preprocess.r

../data/processed/exp2: exp2_preprocess.r preprocess_common.jl
	Rscript exp2_preprocess.r


stats/exp1: exp1_stats.r ../data/processed/exp1
	Rscript exp1_stats.r

figs/exp1: exp1_figs.r ../data/processed/exp1
	Rscript exp1_figs.r --out exp1 --models 'optimal,flexible' 

figs/exp1_alt: exp1_figs.r ../data/processed/exp1
	Rscript exp1_figs.r --out exp1_alt --models 'optimal,empirical_old' 


stats/exp2: exp2_stats.r ../data/processed/exp2
	Rscript exp2_stats.r

figs/exp2: exp2_figs.r ../data/processed/exp2
	Rscript exp2_figs.r --out exp2 --models 'fixed_optimal,flexible' 

figs/exp2_alt: exp2_figs.r ../data/processed/exp2
	Rscript exp2_figs.r --out exp2_alt --models 'fixed_optimal,fixed_empirical_old'

sync:
	rsync --delete-after -av stats/ ~/Papers/meta-memory/stats/
	rsync --delete-after --exclude *.png --exclude tmp* -av figs/ ~/Papers/meta-memory/figs/
	rsync --delete-after --exclude *.png --exclude tmp* -av figs/ ~/Papers/meta-memory/figs/