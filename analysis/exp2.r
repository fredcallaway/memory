suppressPackageStartupMessages(source("setup.r"))
SIZE = 2.5
MAKE_PDF = FALSE
OUT = "exp2_err"

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
    # filter(n_pres >= 2) %>% 
    transmute(trial_id, name, wid, fix_first, duration,
              rel_pretest_accuracy = pretest_accuracy_first - pretest_accuracy_second) %>%
    group_by(trial_id) %>%
    mutate(step_size = if_else(name == "Human", 200, 200)) %>% 
    # mutate(step_size = 200) %>% 
    mutate(n_step = diff(c(0, round(cumsum(duration/step_size))))) %>% 
    uncount(n_step) %>% 
    group_by(trial_id) %>% 
    mutate(time = (step_size/1000)*row_number())

cutoff = df %>% filter(name == "Human") %>% with(quantile(rt, .95, na.rm=T) / 1000)

sum_timecourse = timecourse %>% 
    filter(time < cutoff) %>% 
    participant_means(fix_first, rel_pretest_accuracy, time)

plt_overall = df %>%
    # filter(n_pres >= 2) %>%
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
    filter(presentation != n_pres) %>% 
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

plt_first = nonfinal %>% 
    filter(presentation == 1) %>% 
    group_by(name, wid) %>% mutate(duration = scale(duration, scale=F)) %>% 
    plot_effect(fixated, duration, "Non-Final First", collapse=T) +
    scale_x_continuous(n.breaks=3) +
    labs(x="Pretest Accuracy of First Cue", y="Fixation Duration", colour="Fixation Type")

plt_other = nonfinal %>% 
    filter(presentation > 1) %>% 
    group_by(name, wid) %>% mutate(duration = scale(duration, scale=F)) %>%
    plot_effect(nonfixated, duration, "Non-Final Non-First", collapse=T) +
    labs(x="Pretest Accuracy of Non-Fixated Cue", y="Fixation Duration", colour="Fixation Type")

(plt_first / plt_other) +
    plot_layout(guides = "collect") +
    plot_annotation(tag_levels = 'A') & 
    theme(plot.tag.position = c(0, 1)) &
    scale_colour_manual(values=c(
        "Non-Final First"="#DCBCF8", 
        "Non-Final Non-First"="#AF7BDC"
    ))
savefig("nonfinal_fixations", 3.2, 2)

# %% --------

plt_first = nonfinal %>% 
    filter(presentation == 1) %>% 
    group_by(name, wid) %>% mutate(duration = scale(duration, scale=F)) %>% 
    plot_effect(fixated, duration, "Non-Final First", collapse=F) +
    scale_x_continuous(n.breaks=3) +
    labs(x="Pretest Accuracy of First Cue", y="Z-scored\nFixation Duration", colour="Fixation Type")

plt_other = nonfinal %>% 
    filter(presentation > 1) %>% 
    group_by(name, wid) %>% mutate(duration = scale(duration, scale=F)) %>%
    plot_effect(relative, duration, "Non-Final Non-First", collapse=F) +
    labs(x="Relative Pretest Accuracy of Fixated Cue", y="Z-scored\nFixation Duration", colour="Fixation Type")

(plt_first / plt_other) +
    plot_layout(guides = "collect") +
    plot_annotation(tag_levels = 'A') & 
    theme(plot.tag.position = c(0, 1)) &
    scale_colour_manual(values=c(
        "Non-Final First"="#DCBCF8", 
        "Non-Final Non-First"="#AF7BDC"
    ))
savefig("nonfinal_fixations_alt", 3.2, 2)

# %% --------

plt_first = nonfinal %>% 
    group_by(name, wid) %>% mutate(duration = scale(duration, center=T, scale=F)) %>% 
    plot_effect(fixated, duration, "Non-Final", collapse=T) +
    scale_x_continuous(n.breaks=3) +
    labs(x="Pretest Accuracy of Fixated Cue", y="Fixation Duration", colour="Fixation Type")

plt_other = nonfinal %>% 
    filter(presentation > 1) %>% 
    group_by(name, wid) %>% mutate(duration = scale(duration, center=T, scale=F)) %>%
    plot_effect(nonfixated, duration, "Non-Final Non-Initial", collapse=T) +
    labs(x="Pretest Accuracy of Non-Fixated Cue", y="Fixation Duration", colour="Fixation Type")

(plt_first / plt_other) +
    plot_layout(guides = "collect") +
    plot_annotation(tag_levels = 'A') & 
    theme(plot.tag.position = c(0, 1)) &
    scale_colour_manual(values=c(
        "Non-Final"="#DCBCF8", 
        "Non-Final Non-Initial"="#AF7BDC"
    ))

savefig("nonfinal_fixations_alt2", 3.2, 2)

# %% --------

nonfinal %>% 
    filter(presentation > 1) %>% 
    group_by(name, wid) %>% mutate(duration = scale(duration)) %>%
    plot_effect(relative, duration, "Non-Final Non-First", collapse=T) +
    labs(x="Relative Pretest Accuracy of Fixated Cue", y="Z-scored\nFixation Duration", colour="Fixation Type")

savefig("nonfinal_fixations_relative", 3.2, 1)

# %% ==================== last fixation duration ====================

plt_last_duration = fixations %>% 
    filter(duration <= 5000) %>%
    mutate(type=if_else(last_fix==1, "Final", "Non-Final")) %>% 
    ggplot(aes(duration/1000, fill=type, y = ..width..*..density..)) +
    geom_histogram(position="identity", breaks=seq(0, 5.001, .200), alpha=0.5) +
    facet_grid(~name) +
    # theme(legend.position="None") +
    scale_colour_manual(values=c(
        "Final"="#87DE7A", 
        "Non-Final"="#AF7BDC"
        # "final"="#F8E500", 
        # "non-final"="#17D6CC"
    ), aesthetics=c("fill", "colour"), name="Fixation Type") +
    labs(x="Fixation Duration (s)", y="Proportion")
    # scale_x_continuous(breaks=seq(-1,5,1))

savefig("last_duration", 3, 1)

# %% ====================  ====================
fixations %>% 
    filter(presentation < 5) %>% 
    mutate(final = if_else(n_pres == presentation, "final", "non-final")) %>% 
    plot_effect(presentation, duration, final)

savefig("by_fixation", 3, 1)

# %% ==================== heatmap ====================

nonfinal %>% 
    filter(presentation != n_pres) %>% 
    filter(presentation > 1) %>% 
    # filter(between(duration, -3, 3)) %>% 
    group_by(name, wid) %>% mutate(duration = scale(duration, scale=T)) %>%
    group_by(name, fixated, nonfixated) %>% 
    # filter(n() > 10) %>% 
    summarise(duration=mean(duration, na.rm=T)) %>%
    ggplot(aes(fixated, nonfixated, fill=duration)) +
    geom_tile() +
    facet_wrap(~name)

savefig("heatmap", 3, 1)

 
    # facet_grid(presentation~name, labeller=labeller(presentation=c("1"="First", "2"="Second", "3"="Third")))


