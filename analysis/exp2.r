suppressPackageStartupMessages(source("setup.r"))
S = 2.5
MAKE_PDF = TRUE
write_tex = tex_writer("stats")

# %% ==================== load data ====================

pretest = read_csv('../data/processed/exp2/pretest.csv', col_types = cols())

df = load_model_human("exp2", "trials") %>% 
    filter(response_type == "correct") %>% 
    mutate(rel_pretest_accuracy = pretest_accuracy_first - pretest_accuracy_second)

fixations = load_model_human("exp2", "fixations") %>%
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

# %% ==================== descriptive stats ====================

# write_tex("N/recruited", length(unique(multi_raw$wid)))
# write_tex("N/recruited", length(unique(multi$wid)) + N_drop_acc)
# write_tex("N/drop_acc", N_drop_acc)
# write_tex("N/analysed", length(unique(multi$wid)))

# %% --------

# simple %>% 
#     group_by(wid) %>% 
#     filter(block == max(block)) %>% 
#     filter(n() == 80) %>% 
#     count(response_type) %>% 
#     mutate(prop=prop.table(n)) %>% 
#     group_by(response_type) %>% 
#     summarise(mean=mean(prop), sd=sd(prop)) %>%
#     rowwise() %>% group_walk(~ with(.x, 
#         write_tex("simple_response_pct/{response_type}", "{100*mean:.1}\\% $\\pm$ {100*sd:.1}\\%")
#     ))

# %% --------
# human %>% 
#     filter(n_pres >= 2) %>% 
#     lmer(prop_first ~ rel_pretest_accuracy * last_pres + (rel_pretest_accuracy * last_pres | wid), data=.) %>% 
#     tidy %>% 
#     filter((term != "(Intercept)") & (are_na(effect) | effect == "fixed")) %>% 
#     rowwise() %>% group_walk(~ with(.x,
#         write_tex("overall_interaction/{term}", regression_tex())
#     ))

# %% ==================== overall proportion and timecourse ====================

timecourse = fixations %>% 
    transmute(trial_id, name, fix_first, duration,
              rel_pretest_accuracy = pretest_accuracy_first - pretest_accuracy_second) %>%
    group_by(trial_id) %>%
    mutate(step_size = if_else(name == "Human", 10, 200)) %>% 
    mutate(n_step = diff(c(0, round(cumsum(duration/step_size))))) %>% 
    uncount(n_step) %>% 
    group_by(trial_id) %>% 
    mutate(time = (step_size/1000)*row_number())

cutoff = df %>% filter(name == "Human") %>% with(quantile(rt, .95) / 1000)
sum_timecourse = timecourse %>% 
    filter(time < cutoff) %>% 
    group_by(name, rel_pretest_accuracy, time) %>% 
    summarise(fix_first=mean(fix_first, na.rm=T)) 

# %% --------



plt_overall = df %>%
    filter(n_pres >= 2) %>%
    mutate(x = factor(rel_pretest_accuracy), y = total_first / (total_first + total_second)) %>% 
    ggplot(aes(x, y)) +
    geom_hline(yintercept=0.5, size=.5) +
    stat_summary(fun=mean, group=0, geom="line", colour="#DADADA") +
    stat_summary(aes(color=x), fun.data=mean_cl_normal) +
    facet_wrap(~name) +
    theme(legend.position="none") +
    labs(x="Relative Pretest Accuracy of First Cue", y="Proportion Fixation\nTime on First Cue")

plt_timecourse = sum_timecourse %>% 
    ggplot(aes(time, fix_first, color=factor(rel_pretest_accuracy))) +
    geom_hline(yintercept=0.5, size=.5) +
    geom_line(size=.9) +
    facet_wrap(~name) +
    labs(x="Time (s)", y="Probability Fixate First Cue")


(plt_overall / plt_timecourse) +
    plot_layout(guides = "collect") +
    plot_annotation(tag_levels = 'A') & 
    theme(plot.tag.position = c(0, 1)) &
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
fig("exp2/overall_timecourse", 3.5*S, 2*S)

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

plt_first = nonfinal %>% 
    group_by(name, wid) %>% mutate(duration = scale(duration, center=T, scale=T)) %>% 
    plot_effect(fixated, duration, "Non-Final") +
    scale_x_continuous(n.breaks=3) +
    labs(x="Pretest Accuracy of Fixated Cue", y="Z-scored\nFixation Duration", colour="Fixation Type")

plt_other = nonfinal %>% 
    filter(presentation > 1) %>% 
    group_by(name, wid) %>% mutate(duration = scale(duration, center=T, scale=T)) %>%
    plot_effect(nonfixated, duration, "Non-Final\nNon-Initial") +
    labs(x="Pretest Accuracy of Non-Fixated Cue", y="Z-scored\nFixation Duration", colour="Fixation Type")

(plt_first / plt_other) +
    plot_layout(guides = "collect") +
    plot_annotation(tag_levels = 'A') & 
    theme(plot.tag.position = c(0, 1)) &
    scale_colour_manual(values=c(
        "Non-Final"="#DCBCF8", 
        "Non-Final\nNon-Initial"="#AF7BDC"
        # "Non-Final First"="#17D6CC", 
        # "Non-Final Non-First"="#11A49D"
    ))

fig("exp2/nonfinal_fixations", 3.2*S, 2*S)

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

fig("exp2/last_duration", 3*S, S)

# %% --------

fixations %>% 
    mutate(fix_chosen = choose_first == mod(presentation, 2)) %>% 
    filter(duration <= 5000) %>%
    mutate(type=if_else(fix_chosen, "Chosen", "Non-Chosen")) %>% 
    ggplot(aes(duration/1000, fill=type, y = ..width..*..density..)) +
    geom_histogram(position="identity", breaks=seq(0, 5.001, .200), alpha=0.5) +
    facet_grid(~name) +
    # theme(legend.position="None") +
    scale_colour_manual(values=c(
        "Chosen"="#87DE7A", 
        "Non-Chosen"="#AF7BDC"
        # "final"="#F8E500", 
        # "non-final"="#17D6CC"
    ), aesthetics=c("fill", "colour"), name="Fixation Type") +
    labs(x="Fixation Duration (s)", y="Proportion")
    # scale_x_continuous(breaks=seq(-1,5,1))

fig("exp2/last_duration", 3*S, S)

# %% ==================== Own vs other ====================


X = fixations %>% 
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

X %>% 
    group_by(name,wid) %>% mutate(duration = scale(duration, scale=F)) %>%
    plot_effect(fixated, duration)

fig("tmp", 3*S, S)
# %% --------
X %>% 
    filter(presentation > 1) %>% 
    # group_by(name,wid) %>% mutate(duration = scale(duration, scale=F)) %>%
    plot_effect(nonfixated, duration)
fig("tmp", 3*S, S)

# %% --------

X %>% 
    filter(name=="Human") %>% 
    filter(presentation > 1) %>% 
    with(lmer(duration ~ fixated + nonfixated + (fixated + nonfixated | wid), data=.)) %>% 
    summ

X %>% 
    filter(name=="Human") %>% 
    filter(presentation > 1) %>% 
    with(lmer(duration ~ nonfixated + (nonfixated | wid), data=.)) %>% 
    summ

X %>% 
    filter(name=="Human") %>% 
    with(lmer(duration ~ fixated + (fixated | wid), data=.)) %>% 
    summ


# %% --------

X %>% 
    mutate(
        fixated = midbins(fixated, seq(-3, 3, 3)),
        nonfixated = midbins(nonfixated, seq(-3, 3, 3))
        # fixated = midbins(fixated, seq(-2.5, 2.5, 1)),
        # nonfixated = midbins(nonfixated, seq(-2.5, 2.5, 1))
    ) %>% 
    filter(between(duration, -3, 3)) %>% 
    group_by(name, presentation, fixated, nonfixated) %>% 
    filter(n() > 10) %>% 
    summarise(duration=mean(duration)) %>%
    ggplot(aes(fixated, nonfixated, fill=duration)) +
    geom_tile() + 
    facet_grid(presentation~name, labeller=labeller(presentation=c("1"="First", "2"="Second", "3"="Third")))


