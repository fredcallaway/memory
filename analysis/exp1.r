suppressPackageStartupMessages(source("setup.r"))

pal = scale_colour_manual(values=c(
    'Human'='gray10',
    'Optimal'='#9D6BE0',
    'Random'='gray60'
), aesthetics=c("fill", "colour"), name="") 

pretest = read_csv('../data/processed/exp1/pretest.csv', col_types = cols())
df = load_model_human("exp1", "trials") %>% 
    mutate(
        skip=response_type=="empty", correct=response_type=="correct",
        response_type = recode_factor(response_type, "correct" = "Recalled", "empty" = "Skipped")
    ) %>% 
    group_by(name, wid) %>% 
    mutate(rt_z = zscore(rt)) %>% 
    ungroup()

plot_effect = function(df, x, y) {
    enough_data = df %>% 
        # filter(name == "Human") %>% 
        count(name, response_type, {{x}}) %>% 
        filter(n > 10)

    df %>% 
        right_join(enough_data) %>% 
        ggplot(aes({{x}}, {{y}}, color=name, linetype=name)) +
            stat_summary(fun=mean, geom="line") +
            stat_summary(fun.data=mean_cl_normal, size=.5) +
            theme(legend.position="none") +
            pal +
            scale_linetype_manual(values=c(
                'Human'='solid',
                'Optimal'='dashed',
                'Random'='dashed'
            ))
}

S = 2.5

# %% --------

acc_rt = df %>% plot_effect(pretest_accuracy, rt_z) +
    facet_wrap(~response_type) +
    labs(x="Pretest Accuracy", y='Reaction Time') +
    coord_cartesian(xlim=c(NULL), ylim=c(-1.5, 1.5)) +
    scale_x_continuous(n.breaks=3)

fig("tmp", 2*S, S)

# %% --------

judge_rt = df %>% 
    plot_effect(judgement, rt_z) +
    facet_wrap(~response_type) +
    labs(x="Judgement", y='Reaction Time') +
    coord_cartesian(xlim=c(NULL), ylim=c(-1.5, 1.5)) +
    scale_x_continuous(n.breaks=5)

fig("tmp", 2*S, S)

# %% --------

fast_skip = df %>% 
    filter(!(correct & (rt < 1000))) %>% 
    mutate(y = as.numeric(skip & rt < 1000)) %>%
    plot_effect(pretest_accuracy, y) +
    coord_cartesian(xlim=c(NULL), ylim=c(0, 1)) +
    labs(x="Pretest Accuracy", y="Proportion Fast Skip")

fig("tmp", S, S)

p_skip = df %>% 
    plot_effect(pretest_accuracy, 1*skip) +
    labs(x="Pretest Accuracy", y='Proportion Skip') +
    scale_x_continuous(n.breaks=3)

fig("tmp", S, S)

rt_dist = df %>% ggplot(aes(rt, color=name)) +
    stat_ecdf() +
    facet_wrap(~response_type) +
    theme(legend.position="none") +
    pal + 
    labs(x="Reaction Time", y='Density')

fig("tmp", 2*S, S)

(acc_rt | (p_skip + fast_skip)) / (judge_rt + rt_dist) + plot_annotation(tag_levels = 'A')
fig("stopping_full", 10, 6)

# %% ==================== Metacognitive accuracy ====================
human = df %>% filter(name=="Human")

human %>% 
    plot_effect(pretest_accuracy, judgement) +
    geom_violin(aes(group=pretest_accuracy)) +
    facet_wrap(~response_type) +
    labs(x="Pretest Accuracy", y="Judgement")

fig("tmp", 2*S, S)
# %% --------
human %>% 
    plot_effect(judgement, pretest_accuracy) +
    facet_wrap(~response_type) +
    labs(x="Judgement", y="Pretest Accuracy")

fig("tmp", 2*S, S)

# %% --------
human %>% 
    group_by(wid) %>% 
    mutate(judgement = zscore(judgement), pretest_accuracy=zscore(pretest_accuracy)) %>% 
    ggplot(aes(judgement, pretest_accuracy)) +
    geom_point(size=.1) +
    geom_smooth(method="lm") +
    facet_wrap(~response_type) +
    labs(x="Judgement", y="Pretest Accuracy")

fig("tmp", 2*S, S)
# %% --------

human %>% #plot
    filter(skip) %>% 
    regress(judgement, pretest_accuracy, bins=0, bin_range=1) +
    # stat_summary(fun.data=mean_cl_boot, size=.2) +
    stat_summary(aes(group=wid), fun.y=mean, size=.2, geom="line") +
    xlab("Feeling of Knowing Judgement")
fig()
# %% --------

fok = human %>% 
    filter(skip) %>%
    group_by(wid) %>% 
    filter(sd(judgement) != 0)

X = fok %>% 
    group_modify(function(data, grp) {
        lm(judgement ~ pretest_accuracy, data=data) %>% tidy(conf.int=T)
    }) %>% 
    filter(term == "pretest_accuracy")  %>% 
    ungroup() %>% 
    arrange(estimate) %>%
    mutate(wid=fct_reorder(wid, conf.low))

X %>% 
    ggplot(aes(estimate, wid, xmin=conf.low, xmax=conf.high)) +
    geom_pointrange() + geom_vline(xintercept=0) +
    theme(legend.position="top")

fig()

# %% --------

human %>% 
    filter(skip) %>% 
    ggplot(aes(judgement, ..prop..)) +
    geom_bar() +
    facet_wrap(~pilot) +
    labs(x="Feeling of Knowing Judgement", y="Proportion of Trials")


