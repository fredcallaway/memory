suppressPackageStartupMessages(source("setup.r"))
S = 2.7
MAKE_PDF = TRUE
# %% ==================== load data ====================

pretest = read_csv('../data/processed/exp1/pretest.csv', col_types = cols())
df = load_model_human("exp1", "trials") %>% 
    mutate(
        skip=response_type=="empty", correct=response_type=="correct",
        response_type = recode_factor(response_type, "correct" = "Recalled", "empty" = "Skipped")
    )

# %% ==================== reaction times ====================

acc_rt = df %>%
    plot_effect(pretest_accuracy, rt, response_type) +
    labs(x="Pretest Accuracy", y='Reaction Time') +
    # coord_cartesian(xlim=c(NULL), ylim=c(-1.5, 1.5)) +
    scale_x_continuous(n.breaks=3)

judge_rt = df %>% 
    plot_effect(judgement, rt, response_type) +
    labs(x="Confidence (Recalled) / Feeling of Knowing (Skipped)", y='Reaction Time') +
    # coord_cartesian(xlim=c(NULL), ylim=c(-1.5, 1.5)) +
    scale_x_continuous(n.breaks=5)

(judge_rt / acc_rt) +
    plot_layout(guides = "collect") + 
    plot_annotation(tag_levels = 'A') & 
    theme(
        plot.tag.position = c(0, 1),
        panel.grid.major.y = element_line(color="#EDEDED"),
    ) &
    scale_colour_manual("Response Type", values=c(
        `Skipped`="#DE79AA",
        `Recalled`="#3B77B3"
        # `Recalled`="#3BB365"
    ), aesthetics=c("fill", "colour")) &
    coord_cartesian(xlim=c(NULL), ylim=c(0, 3750))

fig("exp1/rt", 3.5*S, 2*S)


# %% ==================== cummulative probabilities ====================

plot_cum = function(cond, y) {
    seq(0, 5000, 200) %>% 
        map(function(cutoff) {
            df %>% 
                mutate(cutoff = cutoff) %>% 
                filter({{cond}}) %>% 
                mutate(y = as.numeric({{y}}))
        }) %>% 
        bind_rows %>% 
        participant_means(y, cutoff, pretest_accuracy) %>% 
        # group_by(name, cutoff, pretest_accuracy) %>% 
        # summarise(y=mean(y)) %>% 
        mutate(pretest_accuracy=factor(pretest_accuracy)) %>% 
        ggplot(aes(cutoff/1000, y, group=pretest_accuracy)) +
            stat_summary(aes(color=pretest_accuracy), fun=mean, geom="line", size=.9) +
            stat_summary(fun.data=mean_cl_boot, geom="ribbon", alpha=0.08) +
            # geom_line(size=.9) +
            facet_wrap(~name) +
            scale_x_continuous(n.breaks=6)
}

p_correct = plot_cum(TRUE, correct & rt <= cutoff) +
        labs(x="Time (s)", y="Cumulative Recall Probability") +
        scale_colour_manual("Pretest\nAccuracy", values=c(
            `0`="#174675",
            `0.5`="#3B77B3",
            `1`="#4B9AE8"
        ), guide = guide_legend(reverse = TRUE), aesthetics=c("fill", "colour"))

p_skip = plot_cum(!(correct & (rt <= cutoff)), skip & (rt <= cutoff)) +
        labs(x="Time (s)", y="Cummulative Probability of\nSkipping Given No Recall") +
        scale_colour_manual("Pretest\nAccuracy", values=c(
            `0`="#B8648D",
            `0.5`="#DE79AA",
            `1`="#FF92C7"
        ), aesthetics=c("fill", "colour"))

(p_correct / p_skip) +
    plot_annotation(tag_levels = 'A') & 
    theme(plot.tag.position = c(0, 1)) & 
    coord_cartesian(xlim=c(NULL), ylim=c(0, 1))

fig("exp1/cum_probs", 3.5*S, 2.2*S)

# plot_cum(!(correct & (rt <= cutoff)), rt >= cutoff) +
#     labs(x="Time (s)", y="Probability of Continuing\n Search Given No Recall") +
#     scale_colour_manual("Pretest\nAccuracy", values=c(
#         `0`="#B8648D",
#         `0.5`="#DE79AA",
#         `1`="#FF92C7"
#     ), aesthetics=c("fill", "colour"))








