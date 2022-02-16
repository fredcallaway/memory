# %% ==================== Setup ====================

VERSIONS = c('v5.6')

suppressPackageStartupMessages(source("setup.r"))
source("load_exp1.r")
if (!MIXED_DURATIONS) info("Using fixed effects models for fixations")

WIDTH = 5.2; HEIGHT = 2.2
write_tex = tex_writer("stats")

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

multi

# %% ==================== Additional utilities ====================

tidy = function(model, ...) {
    d = broom.mixed::tidy(model, conf.int=T, ...)
    if (typeof(model) == "list") {
        d$df = model$df
    }
    d
}

tidy_models = . %>% 
    rowwise() %>% 
    summarise(tidy(model)) %>% 
    filter((term != "(Intercept)") & (are_na(effect) | effect == "fixed"))

regression_tex = function(logistic=F, standardized=T) {
    beta = if(standardized) "$\\beta = {estimate:.3}$" else "$B = {estimate:.3}$"
    ci = "95\\% CI [{conf.low:.3}, {conf.high:.3}]"
    stat = if(logistic) "$z={statistic:.2}$" else "$t({df:.1})={statistic:.2}$"
    p = "${pval(p.value)}$"
    paste(beta, ci, stat, p, sep=", ")
}

min_strength = -2.5; max_strength = 2.5
breaks_strength = seq(-2.5, 2.5, 1)

run_models = function(data, xvar, yvar, logistic=F) {
    x = ensym(xvar); y = ensym(yvar)
    data %>% 
        group_by(name) %>% 
        group_modify(function(data, grp) {
            model = if (grp$name == "Human") {
                if (logistic) {
                    inject(glmer(!!y ~ !!x + (!!x | wid), family=binomial, data=data))
                } else {
                    inject(lmer(!!y ~ !!x + (!!x | wid), data=data))
                }
            } else {
                if (logistic) {
                    inject(glm(!!y ~ !!x, family=binomial, data=data))
                } else {
                    inject(lm(!!y ~ !!x, data=data))
                }
            }
            tibble(model=list(model))
        })
}

pal = scale_colour_manual(values=c(
    'Human'='gray10',
    'Optimal'='#9D6BE0',
    'Random'='gray60'
), aesthetics=c("fill", "colour"), name="") 

plot_effect = function(df, x, y) {
    ggplot(df, aes({{x}}, {{y}}, color=name, linetype=name)) +
        stat_summary(fun.data=mean_cl_boot, size=.5) +
        stat_summary(fun=mean, geom="line") +
        theme(legend.position="none") +
        pal
}

simple_regression = function(data, xvar, yvar, logistic=F, standardized=T) {
    x = ensym(xvar); y = ensym(yvar)
    models = run_models(data, {{xvar}}, {{yvar}}, logistic)
    models %>% 
        tidy_models() %>% 
        rowwise() %>% group_walk(~ with(.x,
            write_tex("{y}/{name}", regression_tex(logistic, standardized))
        ))

    plot_effect(data, {{xvar}}, {{yvar}})
    # preds = models %>% rowwise() %>% summarise(
        # tibble(ggpredict(model, glue("{x} [n=100]")))
    # )
}


# %% ==================== Sanity check ====================

WIDTH = 3; HEIGHT = 3

df %>% 
    filter(response_type == "correct") %>% 
    mutate(choose_first = int(choose_first)) %>% 
    plot_effect(pre_correct_first, choose_first)

fig("choice", WIDTH, HEIGHT, pdf=T)

df %>% 
    filter(response_type == "correct") %>% 
    filter(n_pres > 0) %>% 
    mutate(
        last_pres_time = map_dbl(presentation_times, last),
        last_pre_correct = if_else(last_pres == "first", pre_correct_first, pre_correct_second)
    ) %>% 
    plot_effect(last_pre_correct, last_pres_time)

fig("last_duration_strength", WIDTH, HEIGHT, pdf=T)

# %% ==================== Overall proportion and timecourse ====================
df %>% filter(name == "Optimal") %>% with(rel_pre_correct)
p_overall = df %>% filter(n_pres >= 2) %>% 
    plot_effect(rel_pre_correct, prop_first)
    # labs(x="Relative Memory Strength of First Cue", y="Proportion Fixation\nTime on First Cue")
fig("prop_first", WIDTH, HEIGHT, pdf=T)

# %% --------

# p_interact = df %>% 
#     filter(n_pres >= 2) %>% 
#     regress_interaction(rel_pre_correct, last_pres, prop_first)

# fig("prop_first_byfinal", WIDTH, HEIGHT)

human %>% 
    filter(n_pres >= 2) %>% 
    lmer(prop_first ~ rel_pre_correct * last_pres + (rel_pre_correct * last_pres | wid), data=.) %>% 
    tidy %>% 
    filter((term != "(Intercept)") & (are_na(effect) | effect == "fixed")) %>% 
    rowwise() %>% group_walk(~ with(.x,
        write_tex("overall_interaction/{term}", regression_tex())
    ))

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

p_time = long %>% 
    normalized_timestep %>% 
    drop_na(strength_diff) %>% 
    ggplot(aes(normalized_timestep/100, fix_stronger, group = strength_diff, color=strength_diff)) +
    geom_smooth(se=F) + 
    ylim(0, 1) +
    facet_grid(~name) +
    labs(x="Trial Completion", y="Probability Fixate\nStronger Cue", color="Strength\nDifference") +
    geom_hline(yintercept=0.5) +
    scale_x_continuous(labels = scales::percent, n.breaks=3)

# fig("normalized-timecourse", WIDTH+1, HEIGHT, pdf=T)

# %% --------

((p_overall + theme(plot.margin= margin(5.5, 5.5, 0, 5.5))) /
 (p_time + theme(plot.margin= margin(0, 5.5, 5.5, 5.5)))) +
    plot_annotation(tag_levels = 'A') & 
    theme(plot.margin=margin(t=1, b=1, l=1, r=1))

fig("overall_and_timecourse", WIDTH+1, HEIGHT*2, pdf=T)

# %% ==================== Fixation Durations ====================

fixdata = long %>% 
    ungroup() %>% 
    filter(presentation < 4 & last_fix == 0) %>% 
    left_join(avg_ptime) %>% 
    mutate(duration = (duration - duration_mean) / duration_sd) %>% 
    left_join(select(df, trial_id, pre_correct_first, pre_correct_second)) %>% 
    mutate(
        fixated = if_else(fix_first==1, pre_correct_first, pre_correct_second),
        nonfixated = if_else(fix_first==1, pre_correct_second, pre_correct_first),
        relative = if_else(fix_first==1, rel_pre_correct, -rel_pre_correct)
    ) %>% 
    pivot_longer(c(fixated, nonfixated, relative), names_to="cue", values_to="pre_correct", names_prefix="") %>% 
    group_by(name, presentation, cue) %>% 
    select(wid, strength, duration)

# %% --------

p1 = df %>% 
    filter(n_pres > 1) %>% 
    plot_effect(pre_correct_first, first_pres_time_z)

p2 = df %>% 
    filter(n_pres > 2) %>% 
    plot_effect(rel_pre_correct, second_pres_time_z)

p3 = df %>% 
    filter(n_pres > 1) %>% 
    plot_effect(rel_pre_correct, third_pres_time_z)

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

fig("fixation_durations_relative", WIDTH, 2.2*HEIGHT, pdf=T)

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

fig("fixation_durations_split", WIDTH+1, 2*HEIGHT, pdf=T)

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

long %>% 
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

fig("last_duration", WIDTH+1.2, HEIGHT, pdf=T)

# %% --------

long %>% 
    filter(duration <= 5000) %>%
    mutate(type=if_else(last_fix==1, "final", "non-final")) %>% 
    simple_regression(last_fix, duration, standardized=F)



