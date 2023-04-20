figs/%: figs/%/exp1 figs/%/exp1_alt figs/%/exp2 figs/%/exp2_alt \
	    figs/%/old_exp1 figs/%/old_exp1_alt figs/%/old_exp2 figs/%/old_exp2_alt
	echo done

stats: stats/exp1 stats/exp2 stats/old_exp1 stats/old_exp2
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

# ---------- large pilot study ----------

stats/old_exp1: exp1_stats.r ../data/processed/old_exp1
	Rscript exp1_stats.r --exp_name old_exp1

figs/%/old_exp1: exp1_figs.r ../data/processed/old_exp1
	Rscript exp1_figs.r --out $@ --run $* --models 'optimal,flexible' --exp_name old_exp1

figs/%/old_exp1_alt: exp1_figs.r ../data/processed/old_exp1
	Rscript exp1_figs.r --out $@ --run $* --models 'optimal,empirical_old' --exp_name old_exp1


stats/old_exp2: exp2_stats.r ../data/processed/old_exp2
	Rscript exp2_stats.r --exp_name old_exp2

figs/%/old_exp2: exp2_figs.r ../data/processed/old_exp2
	Rscript exp2_figs.r --out $@ --run $* --models 'fixed_optimal,flexible'  --exp_name old_exp2

figs/%/old_exp2_alt: exp2_figs.r ../data/processed/old_exp2
	Rscript exp2_figs.r --out $@ --run $* --models 'fixed_optimal,fixed_empirical_old' --exp_name old_exp2


.PRECIOUS: figs/% stats/

