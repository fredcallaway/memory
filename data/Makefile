EXP1_VER = v6.5E
EXP2_VER = v5.6B

processed: processed/exp1 processed/exp2
	
processed/exp1: exp1_preprocess.r common.r
	mkdir -p $@
	Rscript exp1_preprocess.r --version $(EXP1_VER)
	
processed/exp2: exp2_preprocess.r common.r
	mkdir -p $@
	Rscript exp2_preprocess.r --version $(EXP2_VER)