# %% ==================== load data ====================

suppressPackageStartupMessages(source("setup.r"))
suppressPackageStartupMessages(source("stats_base.r"))

pretest = read_csv('../data/processed/exp1/pretest.csv', col_types = cols())
trials = load_human("exp1", "trials") %>% 
    mutate(
        rt = rt / 1000,
        skip=response_type=="empty", correct=response_type=="correct",
        response_type = recode_factor(response_type, "correct" = "Recalled", "empty" = "Skipped")
    )
participants = load_human('exp1', 'participants')

write_tex = tex_writer("stats/exp1")


# %% ==================== accuracy ====================

trials %>% 
    with(mean(correct)) %>% 
    fmt_percent %>% 
    write_tex("accuracy")

# %% ==================== judgement accuracy ====================

trials %>% 
    filter(skip) %>% 
    mutate(rt = scale(rt), pretest_accuracy = scale(pretest_accuracy)) %>% 
    with(lmer(judgement ~ rt + pretest_accuracy + (rt + pretest_accuracy | wid), data=.)) %>% 
    write_model("judgement")

# %% ==================== reaction times ====================

trials %>% 
    filter(skip) %>% 
    regress(rt ~ judgement) %>% 
    write_model("rt_skip1")

trials %>% 
    filter(skip) %>% 
    regress(rt ~ pretest_accuracy) %>%
    write_model("rt_skip2")

trials %>% 
    filter(correct) %>% 
    regress(rt ~ judgement) %>% 
    write_model("rt_correct1")

trials %>% 
    filter(correct) %>% 
    regress(rt ~ pretest_accuracy) %>%
    write_model("rt_correct2")
