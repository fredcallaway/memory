
figs/%: figs/%/exp1 figs/%/exp1_alt figs/%/exp2 figs/%/exp2_alt
	echo done

stats: stats/exp1 stats/exp2
	echo done

stats/exp1: exp1_stats.r ../data/processed/exp1
	Rscript exp1_stats.r

figs/%/exp1: exp1_figs.r ../data/processed/exp1
	Rscript exp1_figs.r --out $@ --run $* --models 'optimal,flexible' 

figs/%/exp1_alt: exp1_figs.r ../data/processed/exp1
	Rscript exp1_figs.r --out $@ --run $* --models 'optimal,empirical_old' 


stats/exp2: exp2_stats.r ../data/processed/exp2
	Rscript exp2_stats.r

figs/%/exp2: exp2_figs.r ../data/processed/exp2
	Rscript exp2_figs.r --out $@ --run $* --models 'fixed_optimal,flexible' 

figs/%/exp2_alt: exp2_figs.r ../data/processed/exp2
	Rscript exp2_figs.r --out $@ --run $* --models 'fixed_optimal,fixed_empirical_old'

.PRECIOUS: figs/%/* stats/