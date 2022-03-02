suppressPackageStartupMessages(source("setup.r"))
S = 2
WIDTH = 5.2; HEIGHT = 2.5; S = 2.7
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
write_tex("N/recruited", length(unique(multi$wid)) + N_drop_acc)
write_tex("N/drop_acc", N_drop_acc)
write_tex("N/analysed", length(unique(multi$wid)))

# %% --------

simple %>% 
    group_by(wid) %>% 
    filter(block == max(block)) %>% 
    filter(n() == 80) %>% 
    count(response_type) %>% 
    mutate(prop=prop.table(n)) %>% 
    group_by(response_type) %>% 
    summarise(mean=mean(prop), sd=sd(prop)) %>%
    rowwise() %>% group_walk(~ with(.x, 
        write_tex("simple_response_pct/{response_type}", "{100*mean:.1}\\% $\\pm$ {100*sd:.1}\\%")
    ))

# %% ==================== overall proportion ====================

overall = df %>% filter(n_pres >= 2) %>% 
    plot_effect(rel_pretest_accuracy, total_first / (total_first + total_second)) +
    # ylim(0, 1) +
    labs(x="Relative Pretest Accuracy of First Cue", y="Proportion Fixation\nTime on First Cue")
fig("exp2/prop_first", 3*S, S)

# %% --------

human %>% 
    filter(n_pres >= 2) %>% 
    lmer(prop_first ~ rel_pretest_accuracy * last_pres + (rel_pretest_accuracy * last_pres | wid), data=.) %>% 
    tidy %>% 
    filter((term != "(Intercept)") & (are_na(effect) | effect == "fixed")) %>% 
    rowwise() %>% group_walk(~ with(.x,
        write_tex("overall_interaction/{term}", regression_tex())
    ))

# %% ==================== timecourse (first) ====================

timecourse = fixations %>% 
    transmute(trial_id, name, fix_first, duration,
              rel_pretest_accuracy = pretest_accuracy_first - pretest_accuracy_second) %>%
    group_by(trial_id) %>%
    # this somewhat complex method ensures that all trials have exactly 100 steps
    # (this isn't true if you just round duration, as I did initially)
    mutate(percentage_complete = round(100*cumsum(duration / sum(duration)))) %>% 
    mutate(n_step = diff(c(0, percentage_complete))) %>% 
    uncount(n_step) %>% 
    group_by(trial_id) %>% 
    mutate(normalized_timestep = row_number())

sum_timecourse = timecourse %>% 
    group_by(name, rel_pretest_accuracy, normalized_timestep) %>% 
    summarise(fix_first=mean(fix_first, na.rm=T)) 

# %% --------

sum_timecourse %>% 
    ggplot(aes(normalized_timestep/100, fix_first, color=factor(rel_pretest_accuracy))) +
    geom_line(size=.8) +
    facet_wrap(~name) +
    labs(x="Trial Completion", y="P(fixate first)") +
    geom_hline(yintercept=0.5) +
    scale_colour_manual(
        values=c(
            "#d7191c",
            "#fdae61",
            "#C0C0C0",
            "#abd9e9",
            "#2c7bb6"
        ), 
        guide = guide_legend(reverse = TRUE), 
        aesthetics=c("colour"),
        name="Relative\nPretest\nAccuracy"
    ) +
    scale_x_continuous(labels = scales::percent, n.breaks=3)

fig("exp2/timecourse_first", 3.5*S, S)

# %% ==================== timecourse (stronger) ====================

timecourse_alt = fixations %>% 
    drop_na(fix_stronger) %>% 
    transmute(trial_id, name, fix_stronger, duration,
              pretest_accuracy_diff = abs(pretest_accuracy_first - pretest_accuracy_second)) %>%
    group_by(trial_id) %>%
    # this somewhat complex method ensures that all trials have exactly 100 steps
    # (this isn't true if you just round duration, as I did initially)
    mutate(percentage_complete = round(100*cumsum(duration / sum(duration)))) %>% 
    mutate(n_step = diff(c(0, percentage_complete))) %>% 
    uncount(n_step) %>% 
    group_by(trial_id) %>% 
    mutate(normalized_timestep = row_number())

sum_timecourse_alt = timecourse_alt %>% 
    group_by(name, pretest_accuracy_diff, normalized_timestep) %>% 
    summarise(fix_stronger=mean(fix_stronger, na.rm=T)) 
# %% --------
sum_timecourse_alt %>% 
    ggplot(aes(normalized_timestep/100, fix_stronger, color=factor(pretest_accuracy_diff))) +
    geom_line(size=.8) +
    facet_wrap(~name) +
    labs(x="Trial Completion", y="P(fixate stronger)") +
    geom_hline(yintercept=0.5) +
    scale_colour_manual("Pretest\nAccuracy\nDifference", values=c(
        `0.5`="dodgerblue4",
        # `0.5`="dodgerblue3",
        `1`="dodgerblue1"
    ), aesthetics=c("fill", "colour")) +
    scale_x_continuous(labels = scales::percent, n.breaks=3)

fig("exp2/timecourse_stronger", 3.5*S, S)


# ((p_overall + theme(plot.margin= margin(5.5, 5.5, 0, 5.5))) /
#  (p_time + theme(plot.margin= margin(0, 5.5, 5.5, 5.5)))) +
#     plot_annotation(tag_levels = 'A') & 
#     theme(plot.margin=margin(t=1, b=1, l=1, r=1))

# fig("overall_and_timecourse", WIDTH+1, HEIGHT*2)

# %% ==================== fixation durations ====================

plt_first = fixations %>% 
    filter(presentation == 1 & presentation != n_pres) %>% 
    group_by(name, wid) %>% mutate(duration = zscore(duration)) %>% 
    plot_effect(pretest_accuracy_first, duration) +
    scale_x_continuous(n.breaks=3) +
    labs(x="Pretest Accuracy of First Cue", y="Normalized Duration of\nNon-Final First Fixations") 

plt_second = fixations %>% 
    filter(presentation > 1) %>% 
    filter(presentation != n_pres) %>% 
    mutate(final = presentation == n_pres) %>% 
    group_by(final,name,wid) %>% mutate(duration = zscore(duration)) %>%
    mutate(
        fixated = case_when(
            mod(presentation, 2) == 1 ~ pretest_accuracy_first,
            mod(presentation, 2) == 0 ~ pretest_accuracy_second,
        ),
        nonfixated = case_when(
            mod(presentation, 2) == 1 ~ pretest_accuracy_second,
            mod(presentation, 2) == 0 ~ pretest_accuracy_first,
        )
    ) %>% mutate(relative = fixated - nonfixated) %>% 
    plot_effect(relative, duration) +
    labs(x="Relative Pretest Accuracy of Fixated Cue", y="Normalized Duration of\nNon-Final Non-Initial Fixations")

(plt_first / plt_second) +
    plot_annotation(tag_levels = 'A') & 
    theme(plot.tag.position = c(0, 1))

fig("exp2/nonfinal_fixations", 3*S, 2.2*S)

# %% ==================== last fixation duration ====================

fixations %>% 
    filter(duration <= 5000) %>%
    mutate(type=if_else(last_fix==1, "final", "non-final")) %>% 
    ggplot(aes(duration/1000, fill=type, y = ..width..*..density..)) +
    geom_histogram(position="identity", breaks=seq(0, 5.001, .200), alpha=0.5) +
    facet_grid(~name) +
    # theme(legend.position="none") +
    scale_colour_manual(values=c(
        "dodgerblue", "gray"
    ), aesthetics=c("fill", "colour"), name="Fixation Type") +
    labs(x="Fixation Duration (s)", y="Proportion")
    # scale_x_continuous(breaks=seq(-1,5,1))

fig("exp2/last_duration", 3*S, S)


# %% ==================== sanity check ====================

choice = df %>% 
    plot_effect(pretest_accuracy_first, int(choose_first))

fig("choice", 3*S, S)

# %% --------

last_duration = df %>% 
    filter(response_type == "correct") %>% 
    filter(n_pres > 0) %>% 
    mutate(
        last_pretest_accuracy = if_else(mod(n_pres, 2) == 1, pretest_accuracy_first, pretest_accuracy_second)
    ) %>% 
    plot_effect(last_pretest_accuracy, last_pres_time)

fig("last_duration_strength", 3*S, S)


