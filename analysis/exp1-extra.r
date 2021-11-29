

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
            tibble(model=model)
        })
}

df %>% 
    filter(n_pres >= 2) %>% 
    regress(rel_strength, prop_first)



# %% --------


regress = function(data, xvar, yvar, bins=6, bin_range=0.95, mixed=TRUE, logistic=FALSE) {
    x = ensym(xvar); y = ensym(yvar)

    xx = filter(data, name == "Human")[[x]]
    q = (1 - bin_range) / 2
    xmin = quantile(xx, q, na.rm=T)
    xmax = quantile(xx, 1 - q, na.rm=T)

    preds = data %>% 
        group_by(name) %>% 
        group_modify(function(data, grp) {
            model = if (grp$name == "Human") {
                if (mixed) {
                    if (logistic) {
                        model = inject(glmer(!!y ~ !!x + (!!x | wid), family=binomial, data=data))
                    } else {
                        model = inject(lmer(!!y ~ !!x + (!!x | wid), data=data))
                    }
                } else {
                    if (logistic) {
                        model = inject(glm(!!y ~ !!x, family=binomial, data=data))
                    } else {
                        model = inject(lm(!!y ~ !!x, data=data))
                    }
                }
                print(glue("N = {nrow(data)}"))
                smart_print(summ(model))
                tibble(ggpredict(model, terms = glue("{x} [n=100]}")))
            } else {
                if (logistic) {
                    model = inject(glm(!!y ~ !!x, family=binomial, data=data))
                } else {
                    model = inject(lm(!!y ~ !!x, data=data))
                }
                tibble(ggpredict(model, terms = glue("{x} [n=100]}")))
            }
        })

    preds %>% 
        filter(between(x, xmin, xmax)) %>% 
        ggplot(aes(x, predicted)) +
        geom_line() +
        geom_ribbon(aes(ymin=conf.low, ymax=conf.high), alpha=0.1) +
        stat_summary_bin(aes({{xvar}}, {{yvar}}), 
            data=data, fun.data=mean_se, bins=5, size=.2, 
            breaks=seq(xmin, xmax, length.out=bins),
        ) +
        facet_grid(~name) +
        labs(x=pretty_name(x), y=pretty_name(y)) +
        coord_cartesian(xlim=c(xmin, xmax))





# %% ==================== Individual Fixation durations ====================

p2 = df %>% 
    filter(n_pres >= 3) %>% 
    mutate(inv_rel_strength = -rel_strength) %>% 
    regress(inv_rel_strength, second_pres_time, mixed=MIXED_DURATIONS) +
    xlab("Second Cue - First Cue Memory Strength")

p3 = df %>% 
    filter(n_pres >= 4) %>% 
    regress(rel_strength, third_pres_time, mixed=MIXED_DURATIONS) + 
    xlab("First Cue - Second Memory Strength") + theme(
      strip.text.x =  element_text(colour = 'white', size=8),
    )

(p2 / p3) +
    expand_limits(y=c(-1, 1)) & 
    theme(plot.margin=margin(t=1, b=1, l=1, r=1))

fig("fixation_durations_relative", WIDTH, 2*HEIGHT, pdf=T)

# %% --------

p1 = df %>%
    filter(n_pres >= 2) %>% 
    regress(strength_first, first_pres_time, mixed=MIXED_DURATIONS)


p2 = df %>% 
    filter(n_pres >= 3) %>% 
    regress(strength_second, second_pres_time, mixed=MIXED_DURATIONS) +
    xlab("Second Cue Memory Strength") + theme(
      strip.text.x =  element_text(colour = 'white', size=8),
    )

p3 = df %>% 
    filter(n_pres >= 4) %>% 
    regress(strength_first, third_pres_time, mixed=MIXED_DURATIONS) + 
    xlab("First Cue Memory Strength") + theme(
      strip.text.x =  element_text(colour = 'white', size=8),
    )

(p1 / p2 / p3) + plot_annotation(tag_levels = 'A') & 
    expand_limits(y=c(-1, 1)) & 
    theme(plot.margin=margin(t=1, b=1, l=1, r=1))

fig("fixation_durations_nonrelative", WIDTH, 3*HEIGHT, pdf=T)
# %% ==================== Old grid ====================


p1 = df %>% 
    filter(n_pres >= 2) %>% 
    group_by(name) %>% 
    group_modify(function(data, grp) {
        model = if (grp$name == "Human") {
            inject(lmer(first_pres_time ~ strength_first + (strength_first|wid), data=data))
        } else {
            inject(lm(first_pres_time ~ strength_first, data=data))
        }
        bind_rows(
            tibble(ggpredict(model, "strength_first [n=100]")) %>% mutate(Cue="Fixated")
        )
    }) %>% 
    plot_each_preds(pretty_name("first_pres_time"))


p2 = df %>% 
    filter(n_pres >= 3) %>% 
    group_by(name) %>% 
    group_modify(function(data, grp) {
        model = if (grp$name == "Human") {
            model = inject(lmer(second_pres_time ~ strength_first + strength_second + 
                (strength_first+strength_second|wid), data=data))
            model
        } else {
            inject(lm(second_pres_time ~ strength_first + strength_second, data=data))
        }
        bind_rows(
            tibble(ggpredict(model, "strength_second [n=100]")) %>% mutate(Cue="Fixated"),
            tibble(ggpredict(model, "strength_first [n=100]")) %>% mutate(Cue="Non-Fixated")
        )
    }) %>% 
    plot_each_preds(pretty_name("second_pres_time"))

p3 = df %>% 
    filter(n_pres >= 4) %>% 
    group_by(name) %>% 
    group_modify(function(data, grp) {
        model = if (grp$name == "Human") {
            model = inject(lmer(third_pres_time ~ strength_first + strength_second + (1|wid), data=data))
            
            model = inject(lmer(third_pres_time ~ strength_second + (strength_second|wid), data=data))
            model = inject(lmer(third_pres_time ~ strength_first + (strength_first|wid), data=data))
            summ(model)   
            # PROBLEM: the mixed effects model gives a stronger effect, can't just drop random slopes
            # maybe we should have separate models for each?
            error("fixme")
            model
        } else {
            inject(lm(third_pres_time ~ strength_first + strength_second, data=data))
        }
        bind_rows(
            tibble(ggpredict(model, "strength_first [n=100]")) %>% mutate(Cue="Fixated"),
            tibble(ggpredict(model, "strength_second [n=100]")) %>% mutate(Cue="Non-Fixated")
        )
    }) %>% 
    plot_each_preds(pretty_name("third_pres_time"))


no_x = theme(axis.title.x=element_text(colour = 'white'))
no_name = theme(strip.text.x =  element_blank())

((p1 + no_x + theme(legend.position = "none")) + 
 (p2 + no_x + no_name) + 
 (p3 + no_name)) + plot_layout(nrow=3, guides="collect") & scale_colour_manual(
    values=c(
        "#57BBF4",
        "#F5D126"
    ), aesthetics=c("fill", "colour")
 )

fig("duration_each_cue", WIDTH+1, 2.6*HEIGHT, pdf=T)


# %% --------

# Fixation duration by participant

fixdata %>% 
    filter(name=="Human") %>% 
    group_by(wid) %>%
    summarise(m=mean(duration), s=sd(duration)) %>% 
    mutate(wid=fct_reorder(wid, m), lo=m-s, hi=m+s) %>% 
    ggplot(aes(wid,m,ymax=hi,ymin=lo)) +
    geom_pointrange()

fig()

# %% --------

fixdata %>% 
    filter(name=="Human") %>% 
    group_by(wid) %>%
    summarise(m=mean(duration), s=sd(duration)) %>% 
    mutate(wid=fct_reorder(wid, m), lo=m-s, hi=m+s) %>% 
    ggplot(aes(wid,m,ymax=hi,ymin=lo)) +
    geom_pointrange()

fig()

# %% ==================== current vs last ====================

human %>%
    filter(n_pres >= 3) %>% 
    regress(strength_first, first_pres_time, mixed=MIXED_DURATIONS)



# %% ==================== Big ass grid ====================
p1 = plot_spacer()

p2 = df %>% 
    filter(n_pres >= 3) %>% 
    mutate(inv_rel_strength = -rel_strength) %>% 
    regress(inv_rel_strength, second_pres_time, mixed=MIXED_DURATIONS) +
    xlab("Second - First Memory Strength") + theme(
      strip.text.x =  element_text(colour = 'white', size=8),
    )

p3 = df %>% 
    filter(n_pres >= 4) %>% 
    regress(rel_strength, third_pres_time, mixed=MIXED_DURATIONS) + 
    xlab("First - Second Memory Strength") + theme(
      strip.text.x =  element_text(colour = 'white', size=8),
    )

a1 = df %>%
    filter(n_pres >= 2) %>% 
    regress(strength_first, first_pres_time, mixed=MIXED_DURATIONS)


a2 = df %>% 
    filter(n_pres >= 3) %>% 
    regress(strength_second, second_pres_time, mixed=MIXED_DURATIONS) +
    xlab("Second Memory Strength")

a3 = df %>% 
    filter(n_pres >= 4) %>% 
    regress(strength_first, third_pres_time, mixed=MIXED_DURATIONS) + 
    xlab("First Memory Strength") + theme(
      strip.text.x =  element_text(colour = 'white', size=8),
    )

((a1 / a2 / a3) | (p1 / p2 / p3)) + plot_annotation(tag_levels = 'A') & 
    expand_limits(y=c(-1, 1)) & 
    theme(plot.margin=margin(t=1, b=1, l=1, r=1))

fig("big_fixation_durations", 2*WIDTH, 3*HEIGHT, pdf=T)

# %% ==================== OG Individual Fixation durations ====================

p1 = df %>%
    filter(n_pres >= 2) %>% 
    regress(strength_first, first_pres_time, mixed=MIXED_DURATIONS)

p2 = df %>% 
    filter(n_pres >= 3) %>% 
    mutate(inv_rel_strength = -rel_strength) %>% 
    regress(inv_rel_strength, second_pres_time, mixed=MIXED_DURATIONS) +
    xlab("Second - First Memory Strength") + theme(
      strip.text.x =  element_text(colour = 'white', size=8),
    )

p3 = df %>% 
    filter(n_pres >= 4) %>% 
    regress(rel_strength, third_pres_time, mixed=MIXED_DURATIONS) + 
    xlab("First - Second Memory Strength") + theme(
      strip.text.x =  element_text(colour = 'white', size=8),
    )

(p1 / p2 / p3) + plot_annotation(tag_levels = 'A') & 
    expand_limits(y=c(-1, 1)) & 
    theme(plot.margin=margin(t=1, b=1, l=1, r=1))

fig("fixation_durations", WIDTH, 3*HEIGHT, pdf=T)


# %% ==================== Comparing Effect of Each Cue ====================

preds = df %>% 
    filter(n_pres >= 2) %>% 
    group_by(name) %>% 
    group_modify(function(data, grp) {
        model = if (grp$name == "Human") {
            inject(lmer(first_pres_time ~ strength_first + (strength_first|wid), data=data))
        } else {
            inject(lm(first_pres_time ~ strength_first, data=data))
        }
        bind_rows(
            tibble(ggpredict(model, "strength_first [n=100]")) %>% mutate(Cue="Fixated")
        )
    })

# %% --------
preds %>% plot_each_preds("Foo")
fig(h=3, w=6) + scale_colour_manual(values=c("Fixated"))


# %% --------

xmin = -3; xmax = 3

plot_each_preds = function(preds, ylab) {    
    preds %>% 
        filter(between(x, xmin, xmax)) %>% 
        ggplot(aes(x, predicted, group=Cue)) +
        geom_ribbon(aes(ymin=conf.low, ymax=conf.high), alpha=0.1) +
        geom_line(aes(color=Cue)) +
        facet_grid(~name) +
        labs(x="Cue Strength", y=ylab) +
        coord_cartesian(xlim=c(xmin, xmax))
}

p1 = df %>% 
    filter(n_pres >= 2) %>% 
    group_by(name) %>% 
    group_modify(function(data, grp) {
        model = if (grp$name == "Human") {
            inject(lmer(first_pres_time ~ strength_first + (strength_first|wid), data=data))
        } else {
            inject(lm(first_pres_time ~ strength_first, data=data))
        }
        bind_rows(
            tibble(ggpredict(model, "strength_first [n=100]")) %>% mutate(Cue="Fixated")
        )
    }) %>% 
    plot_each_preds(pretty_name("first_pres_time"))


p2 = df %>% 
    filter(n_pres >= 3) %>% 
    group_by(name) %>% 
    group_modify(function(data, grp) {
        model = if (grp$name == "Human") {
            model = inject(lmer(second_pres_time ~ strength_first + strength_second + 
                (strength_first+strength_second|wid), data=data))
            model
        } else {
            inject(lm(second_pres_time ~ strength_first + strength_second, data=data))
        }
        bind_rows(
            tibble(ggpredict(model, "strength_second [n=100]")) %>% mutate(Cue="Fixated"),
            tibble(ggpredict(model, "strength_first [n=100]")) %>% mutate(Cue="Non-Fixated")
        )
    }) %>% 
    plot_each_preds(pretty_name("second_pres_time"))

p3 = df %>% 
    filter(n_pres >= 4) %>% 
    group_by(name) %>% 
    group_modify(function(data, grp) {
        model = if (grp$name == "Human") {
            model = inject(lmer(third_pres_time ~ strength_first + strength_second + (1|wid), data=data))
            
            model = inject(lmer(third_pres_time ~ strength_second + (strength_second|wid), data=data))
            model = inject(lmer(third_pres_time ~ strength_first + (strength_first|wid), data=data))
            summ(model)   
            # PROBLEM: the mixed effects model gives a stronger effect, can't just drop random slopes
            # maybe we should have separate models for each?
            error("fixme")
            model
        } else {
            inject(lm(third_pres_time ~ strength_first + strength_second, data=data))
        }
        bind_rows(
            tibble(ggpredict(model, "strength_first [n=100]")) %>% mutate(Cue="Fixated"),
            tibble(ggpredict(model, "strength_second [n=100]")) %>% mutate(Cue="Non-Fixated")
        )
    }) %>% 
    plot_each_preds(pretty_name("third_pres_time"))

# %% --------

no_x = theme(axis.title.x=element_text(colour = 'white'))
no_name = theme(strip.text.x =  element_blank())

((p1 + no_x + theme(legend.position = "none")) + 
 (p2 + no_x + no_name) + 
 (p3 + no_name)) + plot_layout(nrow=3, guides="collect") & scale_colour_manual(
    values=c(
        "#57BBF4",
        "#F5D126"
    ), aesthetics=c("fill", "colour")
 )

fig("duration_each_cue", WIDTH+1, 2.6*HEIGHT, pdf=T)

# %% --------

fig("tmp", WIDTH+1, HEIGHT)

# %% --------

optimal %>% 
    filter(n_pres >= 3) %>% 
    lm(second_pres_time ~ strength_first + strength_second, data=.) %>% 
    summ


human %>% 
    filter(n_pres >= 4) %>% 
    lmer(third_pres_time ~ strength_first + strength_second + (1|wid), data=.) %>% 
    summ

optimal %>% 
    filter(n_pres >= 4) %>% 
    lm(third_pres_time ~ strength_first + strength_second, data=.) %>% 
    summ


# %% --------



# %% alternative version  --------



(p1 / p2 / p3) + plot_annotation(tag_levels = 'A') & 
    expand_limits(y=c(-.8, 1)) & 
    theme(plot.margin=margin(t=1, b=1, l=1, r=1))

fig("fixation_durations_nonrelative", WIDTH, 3*HEIGHT, pdf=T)

# %% ==================== Strength distribution ====================


simple %>%
    filter(block == max(block)) %>% 
    filter(response_type == "correct") %>% 
    filter(rt < 3000) %>% 
    ggplot(aes(rt)) + geom_density()


# %% --------
simple %>%
    filter(block == max(block)) %>% 
    filter(rt < 5000) %>% 
    ggplot(aes(rt, color=response_type=="correct")) + geom_density()

fig(w=7)

# %% --------
compute_strength(block == max(block), 5 * correct - log(rt)) %>%
        ggplot(aes(strength)) + geom_density()

fig()
# %% --------

long %>% 
    normalized_timestep %>% 
    filter(last_fix == 0) %>% 
    drop_na(strength_diff) %>% 
    ggplot(aes(normalized_timestep/100, fix_stronger, group = strength_diff, color=strength_diff)) +
    geom_smooth(se=F) + 
    ylim(0, 1) +
    facet_grid(~name) +
    labs(x="Trial Completion", y="Probability Fixate\nStronger Cue", color="Strength\nDifference") +
    geom_hline(yintercept=0.5) +
    scale_x_continuous(labels = scales::percent, n.breaks=3)

fig("normalized-timecourse-nofinal", WIDTH+1, HEIGHT, pdf=T)

# %% ==================== Unnormalized timecourse ====================

unroll_time = function(long) {
    long %>% 
        group_by(trial_id) %>%
        mutate(n_step = diff(c(0, round(cumsum(duration)/100)))) %>% 
        uncount(n_step) %>%
        group_by(trial_id) %>% 
        mutate(time = 100*row_number())
}

# By strength diff

long %>% 
    unroll_time %>% 
    drop_na(strength_diff) %>% 
    ggplot(aes(time/1000, fix_stronger, group=strength_diff, color=strength_diff)) +
    geom_smooth(se=F) + 
    ylim(0, 1) +
    facet_grid(~name) +
    labs(x="Time in Trial (seconds)", y="Probability Fixate\nStronger Cue", color="Strength\nDifference") +
    geom_hline(yintercept=0.5)


fig("timecourse", WIDTH+1, HEIGHT pdf=T)

# %% ==================== Timecourse by n_pres  ====================

long %>% 
    # filter(name == 'Human') %>% 
    filter(n_pres < 5) %>% 
    left_join(select(human, trial_id, n_pres)) %>% 
    drop_na(strength_diff) %>% 
    filter(strength_diff != "small") %>% 
    unroll_time %>% 
    ggplot(aes(time, fix_stronger, color=factor(n_pres))) +
    geom_smooth(se=F) + 
    ylim(0, 1) +
    facet_grid(~name) +
    labs(x="Time in Trial", y="Probability Fixate\nStronger Cue") +
    geom_hline(yintercept=0.5)


fig("timecourse_bypres", WIDTH+1, HEIGHT, pdf=T)

# # %% --------

# By time

tft_bins = human %>%
    rowwise() %>%
    mutate(total_fix_time = sum(presentation_times)) %>% 
    filter(between(total_fix_time, 0, 5000)) %>% 
    # filter(between(total_fix_time, 1000, 4000)) %>% 
    ungroup() %>% 
    mutate(total_fix_time_bin = cut_width(total_fix_time, 1000, boundary=1000, labels=FALSE)) %>% 
    select(trial_id, n_pres, total_fix_time_bin)

long %>% 
    filter(name == 'Human') %>% 
    # filter(n_pres < 5) %>% 
    right_join(select(tft_bins, trial_id, n_pres, total_fix_time_bin)) %>% 
    drop_na(total_fix_time_bin) %>% 
    unroll_time %>% 
    drop_na(strength_diff) %>% 
    # filter(strength_diff != "small") %>% 
    ggplot(aes(time, fix_stronger, color=factor(total_fix_time_bin))) +
    geom_smooth(se=F) + 
    ylim(0, 1) +
    # facet_grid(~name) +
    labs(x="Time in Trial", y="Probability Fixate Stronger Cue") +
    geom_hline(yintercept=0.5)


fig("timecourse", 4.2, 2.2, pdf=TRUE)
# fig("timecourse", 6.2, 2.2, pdf=T)

# %% --------


# fig("timecourse", 6.2, 2.2, pdf=T)


# %% ==================== All fixations ====================

long %>% 
    filter(last_fix == 0) %>% 
    left_join(select(df, trial_id, strength_first, strength_second)) %>% 
    mutate(strength_fix = if_else(fix_first==1, strength_first, strength_second)) %>% 
    mutate(presentation=factor(presentation)) %>% 
    regress(strength_fix, duration)

fig("nonfinal_strengthfix", WIDTH, HEIGHT)

# %% --------

long %>% 
    filter(last_fix == 0) %>% 
    left_join(select(df, trial_id, strength_first, strength_second)) %>% 
    filter(presentation < 5) %>% 
    mutate(strength_fix = if_else(fix_first==1, strength_first, strength_second)) %>% 
    mutate(presentation=factor(presentation)) %>% 
    regress_interaction(strength_fix, presentation, duration)

fig("nonfinal_strengthfix_bynumber", WIDTH+1, HEIGHT)

# %% --------

long %>% 
    filter(last_fix == 0) %>% 
    filter(between(presentation, 2, 4)) %>% 
    # left_join(select(df, trial_id, strength_first, strength_second)) %>% 
    mutate(rel_strength = if_else(fix_first==1, rel_strength, -rel_strength)) %>% 
    regress(rel_strength, duration)

fig("nonfinal_relative", WIDTH, HEIGHT)

# %% --------

long %>% 
    filter(last_fix == 0) %>% 
    filter(between(presentation, 2, 4)) %>% 
    left_join(avg_ptime) %>% 
    mutate(duration = (duration - duration_mean) / duration_sd) %>%
    mutate(rel_strength = if_else(fix_first==1, rel_strength, -rel_strength)) %>% 
    mutate(presentation=factor(presentation)) %>% 
    regress_interaction(rel_strength, presentation, duration)

fig("nonfinal_relative_bynumber", WIDTH+2, HEIGHT+1)

# %% --------
p2 = df %>% 
    filter(n_pres >= 3) %>% 
    mutate(x = strength_first) %>% 
    regress(x, third_pres_time, mixed=MIXED_DURATIONS) +
    xlab("Second - First Memory Strength") + theme(
      strip.text.x =  element_text(colour = 'white', size=8),
    )
fig()