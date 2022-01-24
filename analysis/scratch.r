
# %% ==================== Debugging  ====================

trials %>% filter(!is.na(response) & str_length(response)==1)  %>% select(version,wid,word,response,judgement)


with(trials, mean(str_length(response)==1, na.rm=T))



# %% ==================== Stopping ====================

pretest %>%
    filter(block==max(block)) %>% 
    group_by(version) %>% 
    count(response_type) %>% 
    ggplot(aes(response_type, n, fill=version)) +
    mutate(n = n/sum(n)) %>% 
    geom_bar(stat="identity", position=position_dodge())

fig()


# %% --------
simple %>% 
    filter(block==3) %>% 
    count(response_type) %>% 
    pivot_wider(names_from=response_type, values_from=n) %>% 
    replace(is.na(.), 0) %>% 
    # select(wid,correct, empty, timeout, intrusion, other) %>% 
    arrange(correct) %>% kable

# %% ==================== Stopping exclusions ====================


# trials = all_trials %>% group_by(wid) %>% filter(mean(correct) > 0.25)
# trials = trials %>%
#     filter(rt > 50) %>%
#     group_by(wid) %>% 
#     filter(n() >= 35) %>% 
#     filter(between(mean(response_type == "empty"), .2, .8))


# n_exclude = length(unique(all_trials$wid)) - length(unique(trials$wid))
# N = length(unique(trials$wid))

# nt = nrow(trials)
# max_rt = with(trials, mean(rt, na.rm=TRUE) + 5 * sd(rt, na.rm=TRUE))
# # trials = filter(trials, rt < max_rt)
# n_trial = nrow(trials)
# n_drop_rt = nt - n_trial

# trials %<>% mutate(
#     log_afc_rt = log(afc_rt),
#     recall_rt = rt,
#     log_recall_rt = log(rt)
# )
# %% ==================== Choose strength ====================

multi = multi %>% add_strength(
    block == max(block), 
    if_else(correct, -log(rt), -10)
)

# multi = multi %>% add_strength(block == max(block), 5 * correct - log(rt))

multi %>% 
    # filter(n_pres == 1) %>% 
    lmer(rt ~ chosen_strength + (chosen_strength|wid), data=.) %>% 
    summ



# %% ==================== Individual regressions ====================

fixdata %>% 
    filter(name == "Human") %>% 
    group_by(presentation, wid) %>% 
    summarise(model = list(lm(duration ~ strength, data=cur_data())))

# %% --------

regs = human %>% 
    filter(n_pres > 1) %>% 
    group_by(wid) %>% 
    filter(n() > 10) %>% 
    summarise(tidy(lm(first_pres_time ~ strength_first, data=cur_data())))


regs %>% 
    filter(term == "strength_first") %>% 
    ggplot(aes(estimate)) + geom_density()
fig()


# %% --------
regs = human %>% 
    filter(n_pres > 2) %>% 
    group_by(wid) %>% 
    filter(n() > 10) %>% 
    summarise(tidy(lm(second_pres_time ~ rel_strength, data=cur_data())))

regs %>% 
    filter(term == "rel_strength") %>% 
    ggplot(aes(estimate)) + geom_density()
fig()

# %% --------


# %% ==================== Fixation Durations ====================

p1 = df %>%
    filter(n_pres >= 2) %>% 
    simple_regression(strength_first, first_pres_time)

p2 = df %>% 
    filter(n_pres >= 3) %>% 
    mutate(inv_rel_strength = -rel_strength)  %>% 
    simple_regression(inv_rel_strength, second_pres_time) +
    xlab("Second Cue Memory Strength - First Cue Memory Strength") + theme(
      strip.text.x =  element_text(colour = 'white', size=8),
    )

p3 = df %>% 
    filter(n_pres >= 4) %>% 
    simple_regression(rel_strength, third_pres_time) + 
    xlab("First Cue Memory Strength - Second Cue Memory Strength") + theme(
      strip.text.x =  element_text(colour = 'white', size=8),
    )


# %% --------
(
    (p1 + theme(plot.margin= margin(5.5, 5.5, 0, 5.5))) / 
    (p2 + theme(plot.margin= margin(0, 5.5, 0, 5.5))) / 
    (p3 + theme(plot.margin= margin(0, 5.5, 5.5, 5.5)))
) + plot_annotation(tag_levels = 'A') & 
    expand_limits(y=c(-1, 2)) & theme(plot.tag.position = c(0, 1))

fig("fixation_durations_nonrelative", WIDTH, 2.7*HEIGHT, pdf=T)

# %% --------

# 2 and 3 only
prep = . %>% filter(cue == "relative" & between(x, -2.5, 2.5) & presentation > 1)

ggplot(prep(preds), aes(x, predicted)) +
    geom_ribbon(aes(ymin=conf.low, ymax=conf.high), alpha=0.1) +
    geom_line() +
    geom_pointrange(aes(y=y, ymin=ymin, ymax=ymax), prep(bindata), size=.2) +
    facet_grid(presentation~name, labeller=labeller(presentation=c("1"="First", "2"="Second", "3"="Third"))) +
    labs(x="Relative Strength of Fixated Cue", y="Fixation Duration")

fig("fixation_durations_relative", WIDTH, 1.6*HEIGHT, pdf=T)
# %% ==================== Heatmaps ====================


# p1 = preds %>% filter(cue == "fixated") %>% group_by(name, presentation) %>% transmute(x1=x, y1=predicted)
# p2 = preds %>% filter(cue == "nonfixated") %>% group_by(name, presentation) %>% transmute(x2=x, y2=predicted)

# %% --------
data = long %>% 
    ungroup() %>% 
    filter(presentation < 4 & last_fix == 0) %>% 
    left_join(avg_ptime) %>% 
    mutate(duration = (duration - duration_mean) / duration_sd) %>% 
    left_join(select(df, trial_id, strength_first, strength_second)) %>% 
    mutate(
        fixated = if_else(fix_first==1, strength_first, strength_second),
        nonfixated = if_else(fix_first==1, strength_second, strength_first),
        relative = if_else(fix_first==1, rel_strength, -rel_strength)
    )

# %% --------

models = data %>% group_by(name, presentation) %>% 
    group_modify(function(data, grp) {
        model = lm(duration ~ fixated + nonfixated, data=data)
        tibble(model=list(model))
    })

preds = models %>% rowwise() %>% summarise(
    tibble(ggpredict(model, c("fixated [-2:2]", "nonfixated [-2:2]")))
)


# %% --------

preds %>% 
    transmute(fixated=x, nonfixated=group, duration=predicted) %>% 
    ggplot(aes(fixated, nonfixated, fill=duration)) + 
    geom_tile() +
    facet_grid(presentation~name, labeller=labeller(presentation=c("1"="First", "2"="Second", "3"="Third")))

fig("fixation_durations_heat2", WIDTH+1, 2*HEIGHT)


# %% --------
models %>% 
    filter(name=="Human" & presentation == 3 & cue == "nonfixated")  %>% 
    with(first(model)) %>% summ

# %% --------
data %>% 
    filter(name=="Human" & presentation == 3) %>% 
    lmer(duration ~ nonfixated + (nonfixated|wid), data=.) %>% summ

data %>% 
    filter(name=="Human" & presentation == 3) %>% 
    lm(duration ~ nonfixated, data=.) %>% summ

# %% ==================== Heatmap with raw data ====================

data %>% 
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

fig("fixation_durations_heat_raw", WIDTH+1, 2*HEIGHT)


# %% --------
summ(model)
model = lm(duration ~ fixated * nonfixated, data=data)
pred = ggpredict(model, terms=c("fixated [-2:2]", "nonfixated [-2:2]")) %>% tibble %>% 
    transmute(fixated=x, nonfixated=group, duration=predicted)

pred %>% ggplot(aes(fixated, nonfixated, fill=duration)) + geom_tile()
fig()

# %% --------
models = data %>% 
    group_modify(function(data, grp) {
        # print(grp)
        model = if (grp$name == "Human") {
            lmer(duration ~ strength + (strength|wid), data=data)
        } else {
            lm(duration ~ strength, data=data)
        }
        tibble(model=list(model))
    })

fixdata


# %% --------

preds %>% 
    filter(cue != "relative") %>% 
    filter(mod(x, 1) == 0) %>% 
    group_by(name, presentation) %>% 
    select(cue, x, predicted) %>% 
    filter(name == "Optimal") %>% print(n=100)

     %>% 
    pivot_wider(names_from=cue, values_from=c(x, predicted))


left_join(p1, p2) %>% 
    mutate(y = y1 + y2) %>% 
    filter(name == "Human" & presentation == 1) %>% 
    ggplot(aes(x1, x2, fill=y)) +
    geom_tile() +

fig()

# %% --------



model = models %>% with(first(model))

ggpredict(model, "strength [-2.5:2.5 by=.1]") %>% tibble

# %% --------

left_join(p1, p2) %>% 
    filter(between(x1, -3, 3) & between(x2, -3, 3)) %>% 
    mutate(y = y1 + y2) %>% 
    filter(name == "Human") %>% 
    ggplot(aes(x1, x2, fill=y)) +
    facet_grid(name ~ presentation) +
    geom_tile()

fig()

# %% --------

X = long %>% 
    filter(presentation < 4 & last_fix == 0) %>% 
    left_join(avg_ptime) %>% 
    mutate(duration = (duration - duration_mean) / duration_sd) %>% 
    left_join(select(df, trial_id, strength_first, strength_second)) %>% 
    mutate(
        fixated = if_else(fix_first==1, strength_first, strength_second),
        nonfixated = if_else(fix_first==1, strength_second, strength_first),
    ) %>% 
    group_by(name, presentation) %>% 
    select(fixated, nonfixated, duration)

X

# %% --------
X %>% 
    filter(name == "Human") %>% 
    filter(presentation == 2) %>% 
    mutate(
        fixated = midbins(fixated, seq(-2.5, 2.5, 1)),
        nonfixated = midbins(nonfixated, seq(-2.5, 2.5, 1))
    ) %>% 
    group_by(fixated, nonfixated) %>% 
    summarise(duration=mean(duration)) %>%
    ggplot(aes(fixated, nonfixated, fill=duration)) +
    geom_tile()

fig()
# %% --------
     %>% 
    mutate(
        # fixated = midbins(fixated, seq(-2.5, 2.5, 1)),
        # nonfixated = midbins(nonfixated, seq(-2.5, 2.5, 1))
    )

     %>% 

    group_by(fixated, nonfixated) %>% 
    summarise(duration=mean(duration)) %>% 
    ggplot(aes(fixated, nonfixated, fill=duration)) +
    geom_tile()

fig()



# %% --------
df %>%
    filter(name=="Human") %>% 
    filter(n_pres >= 2) %>% 
    filter(trial_num >= 10) %>% 
    simple_regression(strength_first, first_pres_time)

# %% --------

compute_strength(block == max(block), 5 * correct - log(rt)) %>% 
    ggplot(aes(raw_strength)) +
    geom_density()
fig()

# %% --------
mean(random$second_pres_time, na.rm=T)

# %% --------
raw_df %>%
    filter(name=="Human") %>% 
    filter(response_type == "other") %>% 
    select(word, response) %>% print(n=100)

raw_df %>%
    filter(name=="Human") %>% 
    filter(response == "no fucking idea")

# %% --------
raw_df %>% 
    filter(name == "Human") %>% 
    group_by(wid) %>% 
    summarise(tibble(rt=quantile(rt, c(0.5, 0.95), na.rm=T), q=c("q50", "q95"))) %>% 
    pivot_wider(names_from=q, values_from=rt) %>% 
    ungroup() %>% 
    mutate(wid = fct_reorder(wid, q50)) %>%
    pivot_longer(c(q50, q95), names_to="name", values_to="value", names_prefix="") %>% 
    ggplot(aes(wid, value, color=name)) +
    geom_point() + scale_x_discrete(breaks=NULL)

fig()

hraw = raw_df %>% filter(name=="Human")

human %>% filter(n_pres <= 6) %>%
    ggplot(aes(rt, color=factor(n_pres))) + geom_density()
fig()

# %% ==================== intrusion by other cue ====================



hraw %>% 
    filter(response_type=="intrusion") %>% 
    filter(n_pres>=2) %>% 
    transmute(
        wid,
        trial_num,
        chosen_word = if_else(choose_first, first_word, second_word),
        unchosen_word = if_else(choose_first, second_word, first_word),
        response
    ) %>% 
    filter(unchosen_word == response)



# %% ==================== Altenrative time courses ====================


unroll_time = function(long) {
    long %>% 
        group_by(trial_id) %>%
        mutate(n_step = diff(c(0, round(cumsum(duration)/100)))) %>% 
        uncount(n_step) %>%
        group_by(trial_id) %>% 
        mutate(time = 100*row_number())
}

long %>% 
    filter(name == 'Human') %>% 
    filter(n_pres < 5) %>% 
    left_join(select(human, trial_id, first_primed, n_pres)) %>% 
    unroll_time %>% 
    drop_na(strength_diff) %>% 
    filter(strength_diff != "small") %>% 
    ggplot(aes(time, fix_stronger, color=as_factor(n_pres))) +
    geom_smooth(se=F) + 
    ylim(0, 1) +
    # facet_grid(~name) +
    labs(x="Time in Trial", y="Probability Fixate Primed Cue") +
    geom_hline(yintercept=0.5) +
    geom_vline(xintercept=1-.707) +
    theme(legend.position="top")

fig()

# %% --------

human %>%
    filter(response_type == "correct") %>% 
    mutate(choose_first = int(choose_first)) %>% 
    regress(strength_first, choose_first, logistic=TRUE) +
    ylab("Prob Select First Cue") + ylim(0, 1) + xlab("First Cue Strength")

fig(w=2.5, h=2.5)

# %% --------

human %>% 
    filter(n_pres >= 2) %>% 
    regress(rel_strength, prop_first) + ylim(0,1)

fig(w=2.5, h=2.5)

# %% --------

human %>% 
    filter(n_pres >= 3) %>% 
    regress(rel_strength, second_pres_time)

fig(w=2.5, h=2.5)

# %% --------

human %>% 
    filter(n_pres >= 4) %>% 
    regress(rel_strength, third_pres_time)

fig(w=2.5, h=2.5)


# %% ==================== Fixation by n fix ====================

long %>% 
    filter(between(n_pres, 2, 5)) %>% 
    left_join(select(human, trial_id, n_pres)) %>% 
    normalized_timestep %>% 
    # drop_na(strength_diff) %>% 
    ggplot(aes(normalized_timestep/100, fix_stronger, color = factor(n_pres))) +
    geom_smooth(se=F) + 
    ylim(0, 1) +
    facet_grid(~name) +
    labs(x="Normalized Time", y="Probability Fixate\nStronger Cue") +
    geom_hline(yintercept=0.5) + 
    theme(legend.position="top") + 
    scale_colour_manual(values=c(
        "#F3F1DD", "#F7DAAF", "#F7AF9D", "#C08497"
    ), aesthetics=c("fill", "colour"), name="")

fig(w=6)

# %% --------


long %>% 
    filter(name == 'Human') %>% 
    left_join(select(human, trial_id, first_primed, n_pres)) %>% 
    mutate(fix_primed = 1*(fix_first == first_primed)) %>% 
    filter(between(n_pres, 1, 4)) %>% 
    normalized_timestep %>% 
    drop_na(strength_diff) %>% 
    ggplot(aes(normalized_timestep/100, fix_primed, color=factor(n_pres))) +
    geom_smooth(se=F) + 
    ylim(0, 1) +
    labs(x="Normalized Time", y="Probability Fixate Primed Cue") +
    geom_hline(yintercept=0.5) +
    theme(legend.position="top") + scale_colour_manual(values=c(
        "#F3F1DD", "#F7DAAF", "#F7AF9D", "#C08497"
    ), aesthetics=c("fill", "colour"), name="")


fig(w=6)


# %% --------
X = long %>% 
    filter(name == 'Human') %>% 
    left_join(select(human, trial_id, first_primed, version))

# %% --------

long %>% 
    filter(name == "Human") %>%
    filter(last_fix == 0) %>% 
    summarise(mean(duration))

# %% --------
human %>% 
    filter(n_pres >= 1) %>% 
    rowwise() %>% 
    mutate(
        last_pres_time = last(presentation_times),
        last_pres_prop = last_pres_time / choice_rt
    ) %>% 
    ungroup() %>% 
    # filter(n_pres < 6) %>% 
    # group_by(n_pres) %>% 
    summarise(median(last_pres_prop))

# %% --------



# %% --------

long %>% 
    filter(name=="Human" & strength_diff  != "small") %>% 
    filter(n_pres > 2) %>% 
    left_join(select(human, trial_id)) %>% 
    normalized_timestep %>% 
    # drop_na(strength_diff) %>% 
    ggplot(aes(normalized_timestep/100, fix_stronger)) +
    geom_smooth(se=F) + 
    ylim(0, 1) +
    facet_grid(~name) +
    labs(x="Normalized Time", y="Probability Fixate\nStronger Cue") +
    geom_hline(yintercept=0.5) + theme(legend.position="right")

fig(w=6)

# %% --------
X = raw_df %>% filter(name == "Human")
# %% --------
X %>% 
    filter(!correct) %>% 
    select()



# %% --------
human %>% 
    filter(n_pres >= 2) %>% 
    add_strength(block == max(block), 5 * correct - log(rt)) %>% 
    lmer(first_pres_time ~ strength_first + (strength_first|wid), data=.) %>% 
    summ

# %% --------

human %>%
    filter(response_type == "correct") %>% 
    add_strength(block == max(block),  7 * correct - log(rt)) %>% 
    mutate(choose_first = int(choose_first)) %>% 
    glmer(choose_first ~ strength_first + (strength_first|wid), data=., family=binomial) %>% 
    summary

# %% --------
load_data('simple-recall') %>% with(table(block))
# %% --------
# Check response coding

```{r}
multi %>% filter(word == response)  %>% with(all(correct)) %>% stopifnot
multi %>% filter(response_type == "other") %>% select(word, response) %>% print(n=100)
multi %>% filter(response_type == "intrusion") %>% select(word, response) %>% print(n=100)
multi %>% filter(response_type == "correct" & word != response) %>% select(word, response) %>% print(n=100)
multi %>% transmute(x=nchar(response)) %>% with(table(x))
multi %>% filter(nchar(response)  == 3) %>% select(response)
multi %>% filter(word == 'ewe') %>% select(word, response, response_type)
```

# %% --------

human %>% 
    filter(n_pres >= 2) %>% 
    lmer(first_pres_time ~ strength_first + (strength_first|wid), data=.) %>% 
    summ

# %% ==================== Pretest Diagnostics ====================

simple %>% ungroup() %>% 
    filter(block == max(block)) %>% 
    summarise(
        mean(rt), sd(rt)
    )

# %% --------
# saveRDS(multi, "/tmp/multi-v5.2")  
# saveRDS(simple, "/tmp/simple-v5.2")
both_simple = bind_rows(
    simple %>% mutate(version = "v5.4"),
    readRDS("/tmp/simple-v5.2") %>% mutate(version = "v5.2")
)
both_multi = bind_rows(
    multi %>% mutate(version = "v5.4"),
    readRDS("/tmp/multi-v5.2") %>% mutate(version = "v5.2")
)
# %% --------
both_simple %>% group_by(version) %>% 
    filter(block == max(block)) %>% 
    summarise(
        mean(rt), sd(rt), 100*mean(correct)
    ) %>% kable(digits=1)
# %% --------
both_simple %>% group_by(block,version) %>% 
    summarise(
        mean(rt), sd(rt), 100*mean(correct)
    ) %>% kable(digits=1)
# %% --------

both_simple %>% ggplot(aes(factor(block), rt, color=version)) +
    geom_boxplot()
fig()

# %% --------

both_simple %>% ggplot(aes(factor(block), rt, color=version)) +
    stat_summary(fun.data=mean_sdl, fun.args = list(mult = 1), position=position_dodge(.2))
fig(w=5)

both_simple %>% ggplot(aes(factor(block), as.numeric(correct), color=version)) +
    stat_summary(fun.data=mean_sdl, fun.args = list(mult = 1), position=position_dodge(.2))
fig(w=5)

# %% ==================== Critical diagnostics ====================
multi52 = readRDS("/tmp/multi-v5.2")
multi$version = "v5.4"
multi52$version = "v5.2"
both = bind_rows(multi, multi52)
# %% --------


both %>% ggplot(aes(rel_strength, as.numeric(choose_first), color=version)) +
    stat_summary_bin(fun.data=mean_cl_boot, bins=5) +
    geom_smooth(method="lm")

fig(w=5)




# %% ==================== GAM / LMER plotting ====================


```{r}

library(mgcv)
library(ggeffects)

X = human %>%
    filter(n_pres >= 2) %>% 
    mutate(log_first_pres_time = log(first_pres_time))

gam_model = gam(first_pres_time ~ s(strength_first, bs = "cs", k = 10) +
                        s(wid, strength_first, bs='re'),
        data=X, method="REML")

# plot1.lin <- ggpredict(m1, terms = c("RPE_mean_abs.s"))
gam_pred <- ggpredict(gam_model, terms = c("strength_first"),
                       condition = c(wid = "new")
                   )
plot(gam_pred)
```

```{r}

lm_model = lm(first_pres_time ~ as.vector(strength_first), data=X)
lm_pred <- ggpredict(lm_model, terms = c("strength_first"), )

lmer_model = lmer(first_pres_time ~ strength_first + (strength_first|wid), data=X)
lmer_pred <- ggpredict(lmer_model, terms = c("strength_first"),
                       condition = c(wid = "new")
                   )
```

```{r}
X %>% ggplot(aes(strength_first)) + 
    geom_line(aes(x, predicted, color="lm"), lm_pred) +
    geom_ribbon(aes(x, ymin=conf.low, ymax=conf.high, fill="lm"), alpha=0.1, lm_pred) +
    geom_line(aes(x, predicted, color="lmer"), lmer_pred) +
    geom_ribbon(aes(x, ymin=conf.low, ymax=conf.high, fill="lmer"), alpha=0.1, lmer_pred) +
    geom_line(aes(x, predicted, color="gam"), gam_pred) +
    geom_ribbon(aes(x, ymin=conf.low, ymax=conf.high, fill="gam"), alpha=0.1, gam_pred)

```

```{r}
p1 <- ggplot() + 
  geom_line(data = plot1.gam, aes(x = x, y = predicted), color = "blue", 
            linetype = "dotted") + 
  geom_ribbon(data = plot1.gam, aes(x = x, ymin = conf.low, 
                                    ymax = conf.high), alpha = 0.1, fill = "blue") +
  geom_line(data = plot1.lin, aes(x = x, y = predicted), color = "black") + 
  geom_ribbon(data = plot1.lin, aes(x = x, ymin = conf.low, 
                                    ymax = conf.high), alpha = 0.2, fill = "black") +
  theme_classic(base_size = 10) + ggtitle("") + xlab("uRPE\n")+ ylab("Curiosity") +  
  stat_summary_bin(data = DataY, aes(x = RPE_mean_abs.s, y = Curiosity_z), 
                   fun.data = "mean_cl_normal", bins = 8, fatten = 0.5) 
p1
```

```{r}
library(mgcv)

X = human %>%
    filter(n_pres >= 2)

m = gam(first_pres_time ~ s(strength_first, bs = "cs", k = 10) +
                        s(wid, strength_first, bs='re'),
        data=X, method="REML")

# bounds = trials %>% summarise(lo=quantile(afc_rt, .05), hi=quantile(afc_rt, .95))
bounds = X %>% group_by(wid) %>% summarise(lo=min(strength_first), hi=max(strength_first))

preds = expand.grid(
    wid = unique(X$wid), 
    strength_first = seq(min(bounds$lo, na.rm=T), max(bounds$hi, na.rm=T), length.out=100)
) %>% mutate(
    first_pres_time=predict(m, newdata=.),
) %>% left_join(bounds) %>% filter(strength_first > lo & strength_first < hi)

preds %>% ggplot(aes(strength_first, first_pres_time, group=wid)) + 
    geom_line(size=.5) +
    theme_classic() +
    coord_fixed(xlim=c(6,9), ylim=c(6,9))


```



# %% ==================== N fixations ====================

human %>% ggplot(aes(n_pres, ..prop..)) + geom_bar()
fig()


# %% ==================== Strength ====================


# RT = xθ / p
# p = xθ / RT
# log(p) = log(xθ) - log(rt)
# p = exp(log(θ) - log(rt))





# %% ==================== Optimal only ====================


```{r}
optimal %>% 
    filter(n_pres >= 2) %>% 
    ggplot(aes(strength_first, first_pres_time)) +
    geom_smooth() + 
    stat_summary_bin(fun.data=mean_cl_boot, bins=5) + 
    labs(x="First Cue Memory Strength", y="First Fixation Time")

optimal %>% 
    filter(n_pres >= 3) %>% 
    ggplot(aes(rel_strength, second_pres_time)) +
    geom_smooth() + 
    stat_summary_bin(fun.data=mean_cl_boot, bins=5) + 
    labs(x="Relative Memory Strength", y="Second Fixation Time")

optimal %>% 
    filter(n_pres >= 4) %>% 
    ggplot(aes(rel_strength, third_pres_time)) +
    geom_smooth() + 
    stat_summary_bin(fun.data=mean_cl_boot, bins=5) + 
    labs(x="Relative Memory Strength", y="Third Fixation Time")
```

# %% --------

make_fixations = function(df) {
    df %>% 
        filter(n_pres >= 1) %>% 
        ungroup() %>% 
        mutate(
            strength_diff = cut(abs(rel_strength), 
                                quantile(abs(rel_strength), c(0, 0.2, 0.6, 1),  na.rm = T),
                                labels=c("small", "moderate", "large"),
                                ordered=T)
        ) %>% 
        mutate(trial = row_number()) %>% 
        unnest_longer(presentation_times, "duration", indices_to="presentation") %>% 
        mutate(
            fix_first = presentation %% 2,
            fix_stronger = as.numeric(fix_first == (rel_strength > 0)),
        )
}

normalized_timestep = function(long) {
    long %>% 
        group_by(trial) %>%
        mutate(prop_duration = duration / sum(duration)) %>% 
        ungroup() %>% 
        mutate(n_step=round(prop_duration * 100)) %>% 
        uncount(n_step) %>% 
        group_by(trial) %>% 
        mutate(normalized_timestep = row_number())
}

timestep = function(long) {
    long %>% 
        mutate(n_step=round(duration / 100)) %>% 
        uncount(n_step) %>% 
        group_by(trial) %>% 
        mutate(timestep = row_number())
}


# %% --------

long = multi %>% make_fixations
long %>% 
    timestep %>% 
    drop_na(strength_diff) %>% 
    ggplot(aes(timestep, fix_stronger)) +
    geom_smooth() + ylim(0, 1) +
    facet_wrap(~phase)
    ggtitle("Human")

fig(w=8)
# %% --------
long %>% 
    normalized_timestep %>% 
    drop_na(strength_diff) %>% 
    ggplot(aes(normalized_timestep, fix_stronger)) +
    geom_smooth() + ylim(0, 1) +
    facet_wrap(~phase)
    ggtitle("Human")
    # theme(legend.position="top")
    
fig(w=8)
# %% --------

optimal %>% 
    ggplot(aes(rt/250)) + geom_histogram()
fig()

# %% --------

long = optimal %>% 
    make_fixations

# %% --------
long %>% 
    mutate(duration = duration * 100) %>% 
    timestep %>%
    drop_na(strength_diff) %>% 
    ggplot(aes(timestep, fix_stronger, group = strength_diff, color=strength_diff)) +
    geom_smooth() +
    ggtitle("Optimal")

fig(w=6)

# %% --------

long %>% 
    normalized_timestep %>% 
    drop_na(strength_diff) %>% 
    ggplot(aes(normalized_timestep, fix_stronger, group = strength_diff, color=strength_diff)) +
    geom_smooth() + ylim(0, 1) +
    ggtitle("Optimal")
    # theme(legend.position="top")
    
fig(w=6)

# %% --------


random %>% 
    make_fixations %>% 
    ggplot(aes(presentation, fix_stronger, color=strength_diff)) + 
    stat_summary(fun.data=mean_cl_boot) + xlim(0, 10)
fig()




# %% ==================== Badness ====================


long %>% 
    filter(presentation == 1) %>% 
    ggplot(aes(abs(rel_strength), fix_stronger)) +
    geom_smooth() + stat_summary_bin(fun.data=mean_cl_boot, bins=5)
fig()
# %% --------

long %>% 
    # filter(strength_diff == "small") %>% 
    ggplot(aes(presentation, fix_stronger, color=strength_diff)) + 
    stat_summary(fun.data=mean_cl_boot)
fig()
# %% --------
long = multi %>% 
    ungroup() %>% 
    mutate(bin_rel_strength = cut(rel_strength, c(-5, -1, 1, 5))) %>% 
    filter(between(n_pres, 1, 5)) %>% 
    select(bin_rel_strength, presentation_times) %>% 
    mutate(trial = row_number()) %>% 
    unnest_longer(presentation_times, "duration", indices_to="presentation") %>% 
    mutate(fix_first = presentation %% 2)

# %% --------
long %>%
    uncount(floor(duration / 100), .id="timestep") %>% 
    select(-duration) %>%
    ggplot(aes(timestep, fix_first, group = bin_rel_strength, color=bin_rel_strength)) +
    geom_smooth()
fig()

# %% ====================  ====================

long %>% 
    group_by(trial) %>%
    mutate(prop_duration = duration / sum(duration)) %>% 
    ungroup() %>% 
    mutate(n_step=round(prop_duration * 100)) %>% 
    uncount(n_step) %>% 
    group_by(trial) %>% 
    mutate(normalized_timestep = row_number()) -> X

X %>% select(normalized_timestep, fix_stronger, strength_diff)
# %% --------
X %>%
    drop_na(strength_diff) %>% 
    ggplot(aes(normalized_timestep, fix_stronger, group = strength_diff, color=strength_diff)) +
    geom_smooth(method=) +
    ggtitle("Human") + 
    theme(legend.position="top")
fig()

# %% --------
long = random %>% 
    ungroup() %>% 
    mutate(bin_rel_strength = cut(rel_strength, c(-5, -1, 1, 5))) %>% 
    filter(between(n_pres, 1, 5)) %>% 
    select(bin_rel_strength, presentation_times) %>% 
    mutate(trial = row_number()) %>% 
    unnest_longer(presentation_times, "duration", indices_to="presentation") %>% 
    mutate(fix_first = presentation %% 2)

long %>% 
    uncount(duration) %>% 
    group_by(trial) %>% 
    mutate(timestep = row_number()) %>% 
    ggplot(aes(timestep, fix_first, group = bin_rel_strength, color=bin_rel_strength)) +
    geom_smooth() +
    theme(legend.position="top")
fig()
# %% --------
random = read_sim("rand_gamma")
random %>% 
    filter(n_pres > 2) %>% 
    ggplot(aes(strength_first, duration_second)) +
    geom_smooth()
fig()


# %% -------- random
long %>% 
    mutate(n_step=duration) %>% 
    uncount(n_step) %>% 
    group_by(trial) %>% 
    mutate(timestep = row_number()) %>% 
    ggplot(aes(timestep, fix_first, group = bin_rel_strength, color=bin_rel_strength)) +
    geom_smooth() +
    theme(legend.position="top")
fig()





# %% --------
long %>%
    uncount(floor(duration / 100), .id="timestep") %>% 
    select(-duration) %>%
    ggplot(aes(timestep, fix_first, group = bin_rel_strength, color=bin_rel_strength)) +
    geom_smooth()
fig()

# %% ==================== First presentation individual slopes ====================


effects = multi %>% 
    filter(n_pres > 1) %>% 
    mutate(fpt_z = scale(first_pres_time)) %>%
    filter(n() > 5) %>% 
    nest(-wid) %>% 
    mutate(
        fit = map(data, ~ 
            lm(fpt_z ~ strength_first, data=.) %>% 
            tidy(conf.int = T)
        )
    ) %>% 
    unnest(fit) %>% 
    filter(term == 'strength_first') %>% 
    arrange(estimate)

ggplot(effects, aes(reorder(wid, estimate), estimate)) + 
    geom_hline(yintercept=0, color="red") +
    geom_point() +
    geom_errorbar(aes(ymin=conf.low , ymax=conf.high)) +
    labs(y="slope(first_pres_time ~ strength_first) [ms/σ]", x="participant") +
    coord_flip() + theme(
        axis.text.y = element_blank(),
        axis.ticks.y = element_blank()
    )

# %% ==================== Linear model weirdness ====================

raw_df = read_sim("rand_gamma")


# %% --------
df = raw_df %>% mutate(
    rel_strength = scale(rel_strength)
    # rel_present = duration_first - duration_second,
    # first_pres_time = map_dbl(presentation_times, 1, .default=NaN),
    # second_pres_time = map_dbl(presentation_times, 2, .default=NaN),
    # choose_first = outcome == 1,
    # n_pres = lengths(presentation_times),
    # odd_pres = mod(n_pres, 2) == 1,
    # rel_strength = scale(strength_first - strength_second),
)


df %>% lm(prop_first ~ rel_strength, data=.) %>% summ
sd(df$rel_strength)


# %% ==================== Fixation time distribution ====================
unnest(multi, presentation_times)

multi$presentation_times


# %% ==================== Easy labels ====================


iris_labs <- iris

## add labels to the columns
lbl <- c('Sepal Length', 'Sepal Width', 'Petal Length', 'Petal Width', 'Flower\nSpecies')
var_label(iris_labs) <- split(lbl, names(iris_labs))

p <- ggplot(iris_labs, aes(x = Sepal.Length, y = Sepal.Width)) +
    geom_line(aes(colour = Species))

p + easy_labs()
fig()
# %% --------
X %>% ggplot(aes(strength_second, first_pres_time)) + 
    stat_summary_bin(fun.data=mean_cl_boot, bins=10) +
    geom_smooth(method='lm')

X %>% lmer(first_pres_time ~ strength_second + (strength_second|wid), data=.) %>% summ
fig()

# %% ==================== Other scoring ====================

X2 = multi %>% 
    filter(n_pres >= 2) %>% 
    add_strength(round > 1, if_else(correct, -logrtz, - 3))

lmer(choose_first ~ strength_first + (strength_first|wid), data=X2) %>% summ
X2 %>% lmer(prop_first ~ rel_strength + (rel_strength|wid), data=.) %>% summ
X2 %>% lmer(second_pres_time ~ strength_first + (strength_first|wid), data=.) %>% summ


# %% ==================== Giving up ====================

trials %>% 
    filter(response_type == "empty") %>% 
    group_by(wid) %>% 
    mutate(
        centered_log_recall_rt = log_recall_rt - mean(log_recall_rt),
        centered_log_afc_rt = log_afc_rt - mean(log_afc_rt),
    ) %>% 
    ungroup() %>% 
    ggplot(aes(centered_log_afc_rt, centered_log_recall_rt, group=wid)) + 
        geom_smooth(method=lm, se=F)

fig()
# %% --------

trials %>% filter(response_type == "other") %>% with(response)

# for simple v4.0 23 is the number of "give up" responses

23 / nrow(trials)



# %% --------


# %% --------
trials %>% ggplot(aes(log_afc_rt, log_recall_rt)) + geom_smooth()
fig()
# %% ==================== Others ====================

simple %>% 
    group_by(wid, word) %>% 
    summarise(n_correct = sum(correct)) %>% 
    ggplot(aes(n_correct)) + geom_bar()
fig()

# %% --------

simple %>% 
    group_by(wid, word) %>% 
    summarise(n_correct = sum(correct)) %>% 
    ungroup() %>% 
    summarise(mean(n_correct == 0))

# %% --------

# This is less likely when the memory strength for the first-seen image is low.

```{r}
ggplot(multi, aes(strength_first, as.numeric(n_pres == 1))) + 
    stat_summary_bin(bins=5) +
    stat_smooth(method="glm", method.args = list(family="binomial")) +
    ylab("p(one presentation)")

glmer(n_pres == 1 ~ strength_first + (1|wid), family='binomial', data=multi) %>% summ
```

afc %>% 
    group_by(wid) %>%
    summarise(mean(correct)) %>% print(n=100)

# %% --------
f = function(data, col) {
    data %>% select({{col}})
}
g = function(data, col) {
    f(data, {{col}})
}

mtcars %>% g(disp)


# %% --------

```{r}
multi %>% 
    ggplot(aes(typing_rt - choice_rt)) + geom_histogram()

multi %>% 
    ggplot(aes(chosen_strength, typing_rt - choice_rt)) + 
    stat_summary_bin(fun.data=mean_cl_boot, bins=5) + 
    geom_smooth(method="lm")
```

# %% --------
multi %>% filter(wid == first(participants$wid)) %>% 
    select(strength_first, strength_second, abs(first_advantage)) %>% 
    arrange(`abs(first_advantage)`) %>% 
    pivot_longer(c(strength_first, strength_second)) %>% 
    ggplot()

# Check javascript score computation
    
check_score <- participants %>%
    select(wid, afc_scores) %>%
    json_to_columns(afc_scores) %>% 
    pivot_longer(-wid, names_to="word") %>% 
    mutate(js_strength = -value) %>% 
    inner_join(afc_scores)

# max(check_score$js_score - check_score$score)
stopifnot(mean(check_score$js_strength - check_score$strength) < .1)



    
