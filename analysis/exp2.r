suppressPackageStartupMessages(source("setup.r"))
SIZE = 2.7
MAKE_PDF = TRUE
STEP_SIZE = 100

RUN = "jun14"
OUT = "exp2"
# OUT = glue("{RUN}_exp2_alt")

savefig = function(name, width, height) {
    fig(glue("{OUT}/{name}"), width*SIZE, height*SIZE, pdf=MAKE_PDF)
}
system(glue('mkdir -p figs/{OUT}'))

# %% ==================== load data ====================
MODELS = c("optimal", "flexible")

pretest = read_csv('../data/processed/exp2/pretest.csv', col_types = cols())
df = load_model_human(RUN, "exp2", "trials") %>% 
    filter(response_type == "correct") %>% 
    mutate(rel_pretest_accuracy = pretest_accuracy_first - pretest_accuracy_second)

fixations = load_model_human(RUN, "exp2", "fixations") %>%
    filter(response_type == "correct") %>% 
    mutate(
        last_fix = as.numeric(presentation == n_pres),
        fix_first = presentation %% 2,
        fix_stronger = case_when(
            pretest_accuracy_first == pretest_accuracy_second ~ NaN,
            pretest_accuracy_first > pretest_accuracy_second ~ 1*fix_first,
            pretest_accuracy_first < pretest_accuracy_second ~ 1*!fix_first
        )
    )

# # %% ==================== overall proportion and timecourse ====================

timecourse = fixations %>% 
    # filter(n_pres >= 2) %>%
    # filter(n_pres <= 3) %>% 
    transmute(trial_id, name, wid, fix_first, duration,
              rel_pretest_accuracy = pretest_accuracy_first - pretest_accuracy_second) %>%
    group_by(trial_id) %>%
    mutate(n_step = diff(c(0, round(cumsum(duration/STEP_SIZE))))) %>% 
    uncount(n_step) %>% 
    group_by(trial_id) %>% 
    mutate(time = (STEP_SIZE/1000)*row_number())

# %% --------

style = list(
    coord_cartesian(xlim=c(NULL), ylim=c(0, 1)),
    scale_colour_manual(
        values=c(
            "#d7191c",
            "#fdae61",
            "#ADADAD",
            "#abd9e9",
            "#2c7bb6"
        ), 
        guide = guide_legend(reverse = TRUE), 
        aesthetics=c("colour"),
        name="Relative\nPretest\nAccuracy"
    )
)

plt_overall = df %>%
    filter(n_pres >= 2) %>%
    mutate(x = factor(rel_pretest_accuracy), y = total_first / (total_first + total_second)) %>% 
    collapse_participants(median, y, x) %>% 
    ggplot(aes(x, y)) +
    stat_summary(fun=mean, group=0, geom="line", colour="#DADADA") +
    stat_summary(aes(color=x), fun.data=mean_cl_boot) +
    facet_wrap(~name) +
    theme(legend.position="none") +
    style +
    labs(x="Relative Pretest Accuracy of First Cue", y="Proportion Fixation\nTime on First Cue")
# savefig("overall", 3.2, 1)

cutoff = df %>% filter(name == "Human") %>% with(quantile(rt, .95, na.rm=T) / 1000)
plt_timecourse = timecourse %>% 
    filter(time < cutoff) %>% 
    plot_effect_continuous(time, fix_first, rel_pretest_accuracy, mean) +
    style +
    labs(x="Time (s)", y="Probability Fixate First Cue")
# savefig("timecourse", 3.5, 1)

(plt_overall / plt_timecourse) +
    plot_layout(guides = "collect") +
    plot_annotation(tag_levels = 'A') & 
    theme(plot.tag.position = c(0, 1))

savefig("overall_timecourse", 3.5, 2)


# %% ==================== fixation durations ====================

nonfinal = fixations %>% 
    filter(last_fix == 0) %>% 
    mutate(type="Non-Final") %>% 
    mutate(
        fixated = case_when(
            mod(presentation, 2) == 1 ~ pretest_accuracy_first,
            mod(presentation, 2) == 0 ~ pretest_accuracy_second,
        ),
        nonfixated = case_when(
            mod(presentation, 2) == 1 ~ pretest_accuracy_second,
            mod(presentation, 2) == 0 ~ pretest_accuracy_first,
        )
    ) %>% mutate(relative = fixated - nonfixated)

# %% --------
plt_fixated = nonfinal %>% 
    mutate(duration=duration/1000)  %>% 
    plot_effect(fixated, duration, type, median) +
    labs(x="Pretest Accuracy of Fixated Cue", y="Duration (s)", colour="Fixation Type")

plt_nonfixated = nonfinal %>% 
    filter(presentation > 1) %>% 
    mutate(duration=duration/1000)  %>% 
    plot_effect(nonfixated, duration, type, median) +
    labs(x="Pretest Accuracy of Non-Fixated Cue", y="Duration (s)", colour="Fixation Type")

(plt_fixated / plt_nonfixated) + 
    plot_layout(guides = "collect") + 
    plot_annotation(tag_levels = 'A') & 
    coord_cartesian(xlim=c(NULL), ylim=join_limits(plt_fixated, plt_nonfixated)) &
    scale_x_continuous(labels = scales::percent, n.breaks=3) &
    scale_colour_manual(values=c(
        # "Final"="#83C57A", 
        "Non-Final"="#AF7BDC"
    ), aesthetics=c("fill", "colour"), name="Fixation Type")

savefig("fixation_durations", 3.5, 2.2)
