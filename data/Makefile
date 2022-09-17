processed: processed/exp1 processed/exp2 processed/old_exp1 processed/old_exp2 
	
processed/exp1: exp1_preprocess.r common.r
	mkdir -p $@
	Rscript exp1_preprocess.r --version v6.5E --exp_name exp1
	
processed/exp2: exp2_preprocess.r common.r
	mkdir -p $@
	Rscript exp2_preprocess.r --version v5.6B --exp_name exp2


processed/old_exp1: exp1_preprocess.r common.r
	mkdir -p $@
	Rscript exp1_preprocess.r --version  v6.5D --exp_name old_exp1
	
processed/old_exp2: exp2_preprocess.r common.r
	mkdir -p $@
	Rscript exp2_preprocess.r --version v5.6 --exp_name old_exp2