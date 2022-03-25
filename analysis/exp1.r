suppressPackageStartupMessages(source("setup.r"))
S = 2.7
MAKE_PDF = TRUE
STEP_SIZE = 100

# %% ==================== load data ====================

# pretest = read_csv('../data/processed/exp1/pretest.csv', col_types = cols())
df = load_model_human("exp1", "trials") %>% 
    mutate(
        skip=response_type=="empty", correct=response_type=="correct",
        response_type = recode_factor(response_type, "correct" = "Recalled", "empty" = "Skipped")
    )

# %% ==================== reaction times ====================

acc_rt = df %>%
    plot_effect(pretest_accuracy, rt, response_type) +
    labs(x="Pretest Accuracy", y='Response Time (s)') +
    # coord_cartesian(xlim=c(NULL), ylim=c(-1.5, 1.5)) +
    scale_x_continuous(labels = scales::percent, n.breaks=3)
    # scale_x_continuous(n.breaks=3)

judge_rt = df %>% 
    plot_effect(judgement, rt, response_type) +
    labs(x="Confidence (Recalled) / Feeling of Knowing (Skipped)", y='Response Time (s)') +
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
    ), aesthetics=c("fill", "colour")) & coord_cartesian(xlim=c(NULL), ylim=c(1000, 3100))

fig("exp1/rt", 3.5*S, 2.2*S)

# %% ==================== cummulative probabilities ====================

cutoff = df %>% 
    filter(name == "Human") %>% 
    with(quantile(rt, .99, na.rm=T)) %>% 
    {STEP_SIZE * round(. / STEP_SIZE)}


plot_cum = function(cond, y) {
    seq(0, cutoff, STEP_SIZE) %>% 
        map(function(cutoff) {
            df %>%
                mutate(cutoff = cutoff) %>% 
                filter({{cond}}) %>% 
                group_by(name, wid, pretest_accuracy, cutoff) %>% 
                summarise(y = mean({{y}}), .groups="keep")
        }) %>% 
        bind_rows %>%
        mutate(x = cutoff / 1000) %>% 
        plot_effect_continuous(x, y, pretest_accuracy) +
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

