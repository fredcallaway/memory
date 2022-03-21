# %% ==================== load data ====================

suppressPackageStartupMessages(source("setup.r"))
suppressPackageStartupMessages(source("stats_base.r"))

pretest = read_csv('../data/processed/exp1/pretest.csv', col_types = cols())
df = load_model_human("exp1", "trials", n=30) %>% 
    mutate(
        rt = rt / 1000,
        skip=response_type=="empty", correct=response_type=="correct",
        response_type = recode_factor(response_type, "correct" = "Recalled", "empty" = "Skipped")
    )
human = df %>% filter(name == "Human")

write_tex = tex_writer("stats/exp1")

# %% ==================== judgement accuracy ====================

human %>% 
    filter(skip) %>% 
    mutate(rt = scale(rt), pretest_accuracy = scale(pretest_accuracy)) %>% 
    with(lmer(judgement ~ rt + pretest_accuracy + (rt + pretest_accuracy | wid), data=.)) %>% 
    write_model("judgement")

# %% ==================== reaction times ====================

human %>% 
    filter(skip) %>% 
    regress(rt ~ judgement) %>% 
    write_model("rt_skip1")

human %>% 
    filter(skip) %>% 
    regress(rt ~ pretest_accuracy) %>%
    write_model("rt_skip2")

human %>% 
    filter(correct) %>% 
    regress(rt ~ judgement) %>% 
    write_model("rt_correct1")

human %>% 
    filter(correct) %>% 
    regress(rt ~ pretest_accuracy) %>%
    write_model("rt_correct2")


# %% --------

df %>% 
    filter(skip) %>% 
    group_by(wid) %>%
    filter(rt < quantile(rt, .95)) %>% 
    mutate(rt = scale(rt)) %>% 
    ggplot(aes(rt, judgement, color=factor(pretest_accuracy))) +
    stat_summary_bin(fun.data=mean_cl_normal, bins=3) +
    facet_wrap(~name)

fig("tmp", 3.5*S, S)


