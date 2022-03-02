suppressPackageStartupMessages(source("setup.r"))

S = 2.5

# %% ==================== load data ====================

pretest = read_csv('../data/processed/exp1/pretest.csv', col_types = cols())
df = load_model_human("exp1", "trials") %>% 
    mutate(
        skip=response_type=="empty", correct=response_type=="correct",
        response_type = recode_factor(response_type, "correct" = "Recalled", "empty" = "Skipped")
    ) %>% 
    group_by(name, wid) %>% 
    mutate(rt_z = zscore(rt)) %>% 
    ungroup()

# %% ==================== reaction times ====================

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

# %% ==================== cummulative probabilities ====================

p_correct = seq(0, 10000, 200) %>% 
    map(function(cutoff) {
        df %>% 
            mutate(
                cutoff=cutoff,
                y = as.numeric(correct & (rt <= cutoff))
            )
    }) %>% 
    bind_rows %>% 
    group_by(name, cutoff, pretest_accuracy) %>% 
    summarise(y=mean(y)) %>% 
    mutate(pretest_accuracy=factor(pretest_accuracy)) %>% 
    ggplot(aes(cutoff/1000, y, color=pretest_accuracy)) +
        geom_line() +
        facet_wrap(~name) +
        scale_x_continuous(n.breaks=6) + 
        labs(x="Time (s)", y="Cumulative Recall Probability") +
        scale_colour_manual("Pretest\nAccuracy", values=c(
            `0`="#174675",
            `0.5`="#3B77B3",
            `1`="#4B9AE8"
        ), guide = guide_legend(reverse = TRUE), aesthetics=c("fill", "colour"))

p_skip = seq(0, 10000, 200) %>% 
    map(function(cutoff) {
        df %>% 
            filter(!(correct & (rt <= cutoff))) %>% 
            mutate(
                cutoff=cutoff,
                y = as.numeric(rt <= cutoff)
            )
    }) %>% 
    bind_rows %>%
    group_by(name, cutoff, pretest_accuracy) %>% 
    summarise(y=mean(y)) %>% 
    mutate(pretest_accuracy=factor(pretest_accuracy)) %>% 
    ggplot(aes(cutoff/1000, y, color=pretest_accuracy)) +
        geom_line() +
        facet_wrap(~name) +
        scale_x_continuous(n.breaks=6) + 
        labs(x="Time (s)", y="Cummulative Probability of\nSkipping Given No Recall") +
        scale_colour_manual("Pretest\nAccuracy", values=c(
            `0`="#B8648D",
            `0.5`="#DE79AA",
            `1`="#FF92C7"
        ), aesthetics=c("fill", "colour"))

(p_correct / p_skip) +
    plot_annotation(tag_levels = 'A') & 
    theme(plot.tag.position = c(0, 1))

fig("exp1/cum_probs", 3.5*S, 2.2*S)
