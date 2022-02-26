# %% ==================== Setup ====================

suppressPackageStartupMessages(source("setup.r"))

WIDTH = 5.2; HEIGHT = 2.5; S = 2.7
write_tex = tex_writer("stats")

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

# %% ==================== Descriptive stats ====================


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

# %% --------




# %% ==================== Sanity check ====================


choice = df %>% 
    plot_effect(pretest_accuracy_first, int(choose_first))

fig("choice", S, S)

# %% --------

last_duration = df %>% 
    filter(response_type == "correct") %>% 
    filter(n_pres > 0) %>% 
    mutate(
        last_pretest_accuracy = if_else(mod(n_pres, 2) == 1, pretest_accuracy_first, pretest_accuracy_second)
    ) %>% 
    plot_effect(last_pretest_accuracy, last_pres_time)

fig("last_duration_strength", S, S)

# %% ==================== Overall proportion and timecourse ====================

overall = df %>% filter(n_pres >= 2) %>% 
    plot_effect(rel_pretest_accuracy, total_first / (total_first + total_second)) +
    ylim(0, 1) +
    labs(x="Relative Pretest Accuracy\nof First Cue", y="Proportion Fixation\nTime on First Cue")
fig("prop_first", S, S)

# %% --------

human %>% 
    filter(n_pres >= 2) %>% 
    lmer(prop_first ~ rel_pretest_accuracy * last_pres + (rel_pretest_accuracy * last_pres | wid), data=.) %>% 
    tidy %>% 
    filter((term != "(Intercept)") & (are_na(effect) | effect == "fixed")) %>% 
    rowwise() %>% group_walk(~ with(.x,
        write_tex("overall_interaction/{term}", regression_tex())
    ))

# %% --------
df %>% 
    group_by(name) %>%
    slice_sample(n=10000) %>% 
    count(name)

# %% --------

normalized_timestep = function(long) {
    long %>% 
        group_by(trial_id) %>%
        # this somewhat complex method ensures that all trials have exactly 100 steps
        # (this isn't true if you just round duration, as I did initially)
        mutate(percentage_complete = round(100*cumsum(duration / sum(duration)))) %>% 
        mutate(n_step = diff(c(0, percentage_complete))) %>% 
        uncount(n_step) %>% 
        group_by(trial_id) %>% 
        mutate(normalized_timestep = row_number())
}

timecourse = fixations %>% 
    normalized_timestep %>% 
    drop_na(fix_stronger) %>% 
    ggplot(aes(normalized_timestep/100, fix_stronger, color=name)) +
    pal + theme(legend.position="none") +
    geom_smooth(se=F) + 
    ylim(0, 1) +
    labs(x="Trial Completion", y="Probability Fixate\nStronger Cue", color="Strength\nDifference") +
    geom_hline(yintercept=0.5) +
    scale_x_continuous(labels = scales::percent, n.breaks=3)

fig("normalized-timecourse", S, S)

# %% --------

nts = fixations %>% 
    normalized_timestep


nts %>% 
    mutate
    group_by()

# %% --------

((p_overall + theme(plot.margin= margin(5.5, 5.5, 0, 5.5))) /
 (p_time + theme(plot.margin= margin(0, 5.5, 5.5, 5.5)))) +
    plot_annotation(tag_levels = 'A') & 
    theme(plot.margin=margin(t=1, b=1, l=1, r=1))

fig("overall_and_timecourse", WIDTH+1, HEIGHT*2)

# %% ==================== Fixation Durations ====================

avg_ptime = fixations %>% group_by(name,wid) %>% 
    filter(last_fix == 0) %>% 
    summarise(duration_mean=mean(duration), duration_sd=sd(duration))

fix1 = df %>% 
    left_join(avg_ptime) %>% 
    filter(n_pres > 1) %>% 
    mutate(first_pres_time = (first_pres_time-duration_mean)/duration_sd) %>% 
    plot_effect(pretest_accuracy_first, first_pres_time)

fix2 = df %>% 
    left_join(avg_ptime) %>% 
    filter(n_pres > 2) %>% 
    mutate(second_pres_time = (second_pres_time-duration_mean)/duration_sd) %>% 
    plot_effect(-rel_pretest_accuracy, second_pres_time)

fix3 = df %>% 
    left_join(avg_ptime) %>% 
    filter(n_pres > 3) %>% 
    mutate(third_pres_time = (third_pres_time-duration_mean)/duration_sd) %>% 
    plot_effect(rel_pretest_accuracy, third_pres_time)

fix1 + fix2 + fix3
fig("fixation_durations", 3*S, S)

# %% --------





# %% ==================== Old ====================


# %% --------
fixdata = long %>% 
    ungroup() %>% 
    filter(presentation < 4 & last_fix == 0) %>% 
    left_join(avg_ptime) %>% 
    mutate(duration = (duration - duration_mean) / duration_sd) %>% 
    left_join(select(df, trial_id, pretest_accuracy_first, pretest_accuracy_second)) %>% 
    mutate(
        fixated = if_else(fix_first==1, pretest_accuracy_first, pretest_accuracy_second),
        nonfixated = if_else(fix_first==1, pretest_accuracy_second, pretest_accuracy_first),
        relative = if_else(fix_first==1, rel_pretest_accuracy, -rel_pretest_accuracy)
    ) %>% 
    pivot_longer(c(fixated, nonfixated, relative), names_to="cue", values_to="pretest_accuracy", names_prefix="") %>% 
    group_by(name, presentation, cue) %>% 
    select(wid, strength, duration)

# %% --------

p1 = df %>% 
    filter(n_pres > 1) %>% 
    plot_effect(pretest_accuracy_first, first_pres_time_z)

p2 = df %>% 
    filter(n_pres > 2) %>% 
    plot_effect(rel_pretest_accuracy, second_pres_time_z)

p3 = df %>% 
    filter(n_pres > 1) %>% 
    plot_effect(rel_pretest_accuracy, third_pres_time_z)

p1 + p2 + p3
fig("fixation_durations", 3*WIDTH, HEIGHT)



# %% --------


models = fixdata %>% 
    group_modify(function(data, grp) {
        # print(grp)
        model = if (grp$name == "Human") {
            lmer(duration ~ strength + (strength|wid), data=data)
        } else {
            lm(duration ~ strength, data=data)
        }
        tibble(model=list(model))
    })

preds = models %>% rowwise() %>% summarise(
    tibble(ggpredict(model, "strength [-2.5:2.5 by=.1]"))
)

bindata = fixdata %>% 
    mutate(x = midbins(strength, seq(-2.5, 2.5, 1))) %>% 
    group_by(x, .add=T) %>%
    summarise(mean_cl_boot(duration))

# %% --------

prep = . %>% filter(
    between(x, -2.5, 2.5) & 
    (presentation > 1 & cue == "relative") |
    (presentation == 1 & cue == "fixated")
)

ggplot(prep(preds), aes(x, predicted)) +
    geom_ribbon(aes(ymin=conf.low, ymax=conf.high), alpha=0.1) +
    geom_line() +
    geom_pointrange(aes(y=y, ymin=ymin, ymax=ymax), prep(bindata), size=.2) +
    facet_grid(presentation~name, labeller=labeller(presentation=c("1"="First", "2"="Second", "3"="Third"))) +
    labs(x="(Relative) Strength of Fixated Cue", y="Standardized Fixation Duration")

fig("fixation_durations_relative", WIDTH, 2.2*HEIGHT)

# %% --------

prep = . %>% 
    filter(cue != "relative" & between(x, -2.5, 2.5)) %>% 
    # filter(!(presentation == 1 & cue == "nonfixated")) %>% 
    mutate(Cue=factor(cue, c("fixated", "nonfixated"), c("Fixated", "Non-Fixated")))

ggplot(prep(preds), aes(x, predicted, group=Cue, color=Cue)) +
    geom_ribbon(aes(ymin=conf.low, ymax=conf.high), alpha=0.1, color=F) +
    geom_line() +
    geom_pointrange(aes(y=y, ymin=ymin, ymax=ymax), prep(bindata), size=.2) +
    facet_grid(presentation~name, labeller=labeller(presentation=c("1"="First", "2"="Second", "3"="Third"))) +
    labs(x="Cue Strength", y="Fixation Duration") + scale_colour_manual(
    values=c(
        "#57BBF4",
        "#F5D126"
    ), aesthetics=c("fill", "colour")
 )

fig("fixation_durations_split", WIDTH+1, 2*HEIGHT)

# %% --------
# model1 = lm(duration ~ strength, data=filter(fixdata, name=='Human'))
# model2 = lmer(duration ~ strength + (strength|wid), data=filter(fixdata, name=='Human'))

models %>%
    filter(name=="Human") %>% 
    rowwise() %>%
    summarise(tidy(model))  %>% 
    filter(term == "strength") %>% 
    rowwise() %>% group_walk(~ with(.x, 
        write_tex("durations/{presentation}/{cue}", regression_tex())
    ))

# %% ==================== Last fixation duration ====================

fixations %>% 
    filter(duration <= 5000) %>%
    mutate(type=if_else(last_fix==1, "final", "non-final")) %>% 
    ggplot(aes(duration/1000, fill=type, y = ..width..*..density..)) +
    geom_histogram(position="identity", breaks=seq(0, 5.001, .250), alpha=0.5) +
    facet_grid(~name) +
    # theme(legend.position="none") +
    scale_colour_manual(values=c(
        "dodgerblue", "gray"
    ), aesthetics=c("fill", "colour"), name="Fixation Type") +
    labs(x="Fixation Duration (s)", y="Proportion")
    # scale_x_continuous(breaks=seq(-1,5,1))

fig("last_duration", WIDTH+1.2, HEIGHT)

# %% --------

long %>% 
    filter(duration <= 5000) %>%
    mutate(type=if_else(last_fix==1, "final", "non-final")) %>% 
    simple_regression(last_fix, duration, standardized=F)



