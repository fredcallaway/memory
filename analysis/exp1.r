suppressPackageStartupMessages(source("setup.r"))

SIZE = 2.7
MAKE_PDF = TRUE
STEP_SIZE = 100

RUN = "may18_fixedthresh"
OUT = "exp1"

savefig = function(name, width, height) {
    fig(glue("{OUT}/{name}"), width*SIZE, height*SIZE, pdf=MAKE_PDF)
}
system(glue('mkdir -p figs/{OUT}'))

# %% ==================== load data ====================

# pretest = read_csv('../data/processed/exp1/pretest.csv', col_types = cols())
df = load_model_human(RUN, "exp1", "trials") %>% 
    mutate(
        skip=response_type=="empty", correct=response_type=="correct",
        response_type = recode_factor(response_type, "correct" = "Recalled", "empty" = "Skipped")
    )

# %% ==================== reaction times ====================


style = list(
    scale_colour_manual("Response Type", values=c(
        `Skipped`="#DE79AA",
        `Recalled`="#3B77B3"
    ), aesthetics=c("fill", "colour")),
    coord_cartesian(xlim=c(NULL), ylim=c(1000, 3500))
)

acc_rt = df %>%
    plot_effect(pretest_accuracy, rt, response_type, median) +
    labs(x="Pretest Accuracy", y='Response Time (s)') +
    scale_x_continuous(labels = scales::percent, n.breaks=3) +
    style
# savefig("acc_rt", 3.5, 1.1)

judge_rt = df %>% 
    plot_effect(judgement, rt, response_type, median) +
    labs(x="Confidence (Recalled) / Feeling of Knowing (Skipped)", y='Response Time (s)') +
    scale_x_continuous(n.breaks=5) + style
# savefig("judge_rt", 3.5, 1.1)

(judge_rt / acc_rt) +
    plot_layout(guides = "collect") + 
    plot_annotation(tag_levels = 'A') & 
    theme(plot.tag.position = c(0, 1))

savefig("rt", 3.5, 2.2)

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
        plot_effect_continuous(x, y, pretest_accuracy, mean) +
        scale_x_continuous(n.breaks=6)
}

p_correct = plot_cum(TRUE, correct & rt <= cutoff) +
        labs(x="Time (s)", y="Cumulative Recall Probability") +
        scale_colour_manual("Pretest\nAccuracy", values=c(
            `0`="#174675",
            `0.5`="#3B77B3",
            `1`="#4B9AE8"
        ), guide = guide_legend(reverse = TRUE), aesthetics=c("fill", "colour"))
savefig("p_correct", 3.5, 1.1)

p_skip = plot_cum(!(correct & (rt <= cutoff)), skip & (rt <= cutoff)) +
        labs(x="Time (s)", y="Cummulative Probability of\nSkipping Given No Recall") +
        scale_colour_manual("Pretest\nAccuracy", values=c(
            `0`="#B8648D",
            `0.5`="#DE79AA",
            `1`="#FF92C7"
        ), aesthetics=c("fill", "colour"))
# savefig("p_skip", 3.5, 1.1)

(p_correct / p_skip) +
    plot_annotation(tag_levels = 'A') & 
    theme(plot.tag.position = c(0, 1)) & 
    coord_cartesian(xlim=c(NULL), ylim=c(0, 1))

savefig("cum_probs", 3.5, 2.2)

# %% --------

plot_cum(TRUE, skip & rt <= cutoff) +
        labs(x="Time (s)", y="Cummulative Skip Probability") +
        scale_colour_manual("Pretest\nAccuracy", values=c(
            `0`="#B8648D",
            `0.5`="#DE79AA",
            `1`="#FF92C7"
        ), aesthetics=c("fill", "colour"))
savefig("p_skip_simple", 3.5, 1.1)

# %% ==================== metacognitive accuracy ====================

pskip = df %>% 
    filter(name == "Human") %>% 
    filter(response_type == "Skipped") %>% 
    plot_effect(pretest_accuracy, judgement, response_type) +
    theme(legend.position="none") +
    labs(x="Pretest Accuracy", y="Feeling of Knowing")

prec = df %>% 
    filter(name == "Human") %>% 
    filter(response_type == "Recalled") %>% 
    plot_effect(pretest_accuracy, judgement, response_type) +
    labs(x="Pretest Accuracy", y="Confidence")

(pskip + prec) &
    plot_layout(guides = "collect") &
    scale_colour_manual("Response Type", values=c(
        `Skipped`="#DE79AA",
        `Recalled`="#3B77B3"
        # `Recalled`="#3BB365"
    ), aesthetics=c("fill", "colour")) &
    theme(strip.text.x=element_blank()) &
    scale_x_continuous(breaks=c(0, 0.5, 1)) &
    coord_cartesian(xlim=c(0, 1), ylim=c(1, 5))

savefig("accuracy", 2.5, 1)


