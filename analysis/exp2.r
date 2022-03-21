suppressPackageStartupMessages(source("setup.r"))
SIZE = 2.5
MAKE_PDF = TRUE
OUT = "exp2"

savefig = function(name, width, height) {
    fig(glue("{OUT}/{name}"), width*SIZE, height*SIZE, pdf=MAKE_PDF)
}
system(glue('mkdir -p figs/{OUT}'))

# %% ==================== load data ====================

pretest = read_csv('../data/processed/exp2/pretest.csv', col_types = cols())

filt = switch(OUT, 
    exp2_err = quote(response_type != "intrusion"), 
    quote(response_type == "correct")
)

df = load_model_human("exp2", "trials") %>% 
    filter(eval(filt)) %>% 
    mutate(rel_pretest_accuracy = pretest_accuracy_first - pretest_accuracy_second)

fixations = load_model_human("exp2", "fixations") %>%
    filter(eval(filt)) %>% 
    mutate(
        last_fix = as.numeric(presentation == n_pres),
        fix_first = presentation %% 2,
        fix_stronger = case_when(
            pretest_accuracy_first == pretest_accuracy_second ~ NaN,
            pretest_accuracy_first > pretest_accuracy_second ~ 1*fix_first,
            pretest_accuracy_first < pretest_accuracy_second ~ 1*!fix_first
        )
    )

# %% ==================== overall proportion and timecourse ====================

timecourse = fixations %>% 
    filter(n_pres >= 2) %>%
    transmute(trial_id, name, wid, fix_first, duration,
              rel_pretest_accuracy = pretest_accuracy_first - pretest_accuracy_second) %>%
    group_by(trial_id) %>%
    mutate(step_size = if_else(name == "Human", 50, 50)) %>% 
    mutate(n_step = diff(c(0, round(cumsum(duration/step_size))))) %>% 
    uncount(n_step) %>% 
    group_by(trial_id) %>% 
    mutate(time = (step_size/1000)*row_number())

cutoff = df %>% filter(name == "Human") %>% with(quantile(rt, .95, na.rm=T) / 1000)

sum_timecourse = timecourse %>% 
    filter(time < cutoff) %>% 
    participant_means(fix_first, rel_pretest_accuracy, time)

plt_overall = df %>%
    filter(n_pres >= 2) %>%
    mutate(x = factor(rel_pretest_accuracy), y = total_first / (total_first + total_second)) %>% 
    ggplot(aes(x, y)) +
    # geom_hline(yintercept=0.5, size=.5) +
    stat_summary(fun=mean, group=0, geom="line", colour="#DADADA") +
    stat_summary(aes(color=x), fun.data=mean_cl_boot) +
    facet_wrap(~name) +
    theme(legend.position="none") +
    labs(x="Relative Pretest Accuracy of First Cue", y="Proportion Fixation\nTime on First Cue")

plt_timecourse = sum_timecourse %>% 
    ggplot(aes(time, fix_first, group=factor(rel_pretest_accuracy))) +
    # geom_hline(yintercept=0.5, size=.5) +
    stat_summary(aes(color=factor(rel_pretest_accuracy)), fun=mean, geom="line", size=.9) +
    # stat_summary(fun.data=mean_cl_boot, geom="ribbon", alpha=0.08) +
    # geom_line(size=.9) +
    facet_wrap(~name) +
    labs(x="Time (s)", y="Probability Fixate First Cue")

(plt_overall / plt_timecourse) +
    plot_layout(guides = "collect") +
    plot_annotation(tag_levels = 'A') & 
    theme(plot.tag.position = c(0, 1)) &
    coord_cartesian(xlim=c(NULL), ylim=c(0, 1)) &
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

plt_number = fixations %>% 
    filter(presentation < 6) %>% 
    mutate(type=if_else(last_fix==1, "Final", "Non-Final")) %>% 
    plot_effect(presentation, duration, type) +
    labs(x="Fixation Number", y="Duration (s)")

plt_fixated = nonfinal %>% 
    # group_by(name, wid) %>% mutate(duration = scale(duration, center=T, scale=F)) %>% 
    plot_effect(fixated, duration, type) +
    scale_x_continuous(n.breaks=3) +
    labs(x="Pretest Accuracy of Fixated Cue", y="Duration (s)", colour="Fixation Type")

plt_nonfixated = nonfinal %>% 
    filter(presentation > 1) %>% 
    # group_by(name, wid) %>% mutate(duration = scale(duration, center=T, scale=F)) %>%
    plot_effect(nonfixated, duration, type) +
    scale_x_continuous(n.breaks=3) +
    labs(x="Pretest Accuracy of Non-Fixated Cue", y="Duration (s)", colour="Fixation Type")

(plt_number / plt_fixated / plt_nonfixated) +
    plot_layout(guides = "collect") +
    plot_annotation(tag_levels = 'A') & 
    theme(plot.tag.position = c(0, 1)) &
    scale_colour_manual(values=c(
        "Final"="#89D37F", 
        "Non-Final"="#AF7BDC"
        # "final"="#F8E500", #8FC786
        # "non-final"="#17D6CC"
    ), aesthetics=c("fill", "colour"), name="Fixation Type")

savefig("fixation_durations", 3.2, 2.5)

# %% ==================== last fixation duration ====================


