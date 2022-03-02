suppressPackageStartupMessages(source("setup.r"))

S = 2.5

# %% ==================== Load data ====================

pretest = read_csv('../data/processed/exp1/pretest.csv', col_types = cols())
df = load_model_human("exp1", "trials") %>% 
    mutate(
        skip=response_type=="empty", correct=response_type=="correct",
        response_type = recode_factor(response_type, "correct" = "Recalled", "empty" = "Skipped")
    ) %>% 
    group_by(name, wid) %>% 
    mutate(rt_z = zscore(rt)) %>% 
    ungroup()

rtyp_pal = scale_colour_manual("Response Type", values=c(
    `Skipped`="#DE79AA",
    `Recalled`="#3B77B3"
    # `Recalled`="#3BB365"
), aesthetics=c("fill", "colour"))

acc_rt = df %>%
    plot_effect(pretest_accuracy, rt, response_type) +
    labs(x="Pretest Accuracy", y='Reaction Time') +
    # coord_cartesian(xlim=c(NULL), ylim=c(-1.5, 1.5)) +
    scale_x_continuous(n.breaks=3) + rtyp_pal

judge_rt = df %>% 
    plot_effect(judgement, rt, response_type) +
    labs(x="Metamemory Judgement", y='Reaction Time') +
    # coord_cartesian(xlim=c(NULL), ylim=c(-1.5, 1.5)) +
    scale_x_continuous(n.breaks=5) + rtyp_pal

(judge_rt / acc_rt) +
    plot_layout(guides = "collect") + 
    plot_annotation(tag_levels = 'A') & 
    theme(plot.tag.position = c(0, 1)) &
    coord_cartesian(xlim=c(NULL), ylim=c(1000, 3600))

fig("exp1/rt", 3.5*S, 2*S)

