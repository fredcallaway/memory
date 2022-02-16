suppressPackageStartupMessages(source("setup.r"))

# source('exp1_preprocess.r')  # so fast, might as well
pretest = read_csv('../data/processed/exp1/pretest.csv', col_types = cols())
trials = read_csv('../data/processed/exp1/trials.csv', col_types = cols())

pal = scale_colour_manual(values=c(
    'Human'='gray10',
    'Optimal'='#9D6BE0',
    'Random'='gray60'
), aesthetics=c("fill", "colour"), name="") 

# %% --------



# %% --------

read_sim  = . %>% 
    read_csv() %>% 
    mutate(response_type = if_else(outcome == 1, "correct", "empty")) %>% 
    # mutate(rt = rt / 1000) %>% 
    mutate(μ_post = μ_post + rnorm(n(), sd=1)) %>% 
    mutate(rt_z = zscore(rt)) %>% 
    mutate(judgement=cut(μ_post, quantile(μ_post, make_breaks(.35)), labels=F))

df = bind_rows(
    read_sim('../model/results/exp1_random.csv') %>% mutate(name="Random"),
    read_sim('../model/results/exp1_optimal.csv') %>% mutate(name="Optimal"),
    trials %>% mutate(name = "Human")
) %>% mutate(
    name = factor(name, levels=c("Random", "Optimal", "Human"), ordered=T),
    correct = response_type == "correct",
    skip = response_type == "empty",

)

plot_effect = function(df, x, y) {
    ggplot(df, aes({{x}}, {{y}}, color=name, linetype=name)) +
        stat_summary(fun=mean, geom="line") +
        stat_summary(fun.data=mean_cl_boot, size=.5) +
        theme(legend.position="none") +
        pal +
        scale_linetype_manual(values=c(
            'Human'='solid',
            'Optimal'='dashed',
            'Random'='dashed'
        ))
}
```

```{r}
df %>% ggplot(aes(pre_correct, rt, color=name)) + stat_summary(fun.data=mean_cl_normal) + pal

acc_rt = df %>% plot_effect(pre_correct, rt) +
    facet_wrap(~response_type) +
    labs(x="Pretest Accuracy", y='Reaction Time') +
    coord_cartesian(xlim=c(NULL), ylim=c(0, 4000)) +
    scale_x_continuous(n.breaks=3)

judge_rt = df %>% plot_effect(judgement, rt) +
    facet_wrap(~response_type) +
    labs(x="Judgement", y='Reaction Time') +
    coord_cartesian(xlim=c(NULL), ylim=c(0, 4000))
    scale_x_continuous(n.breaks=5)

fast_skip = df %>% 
    filter(!(correct & (rt < 1000))) %>% 
    mutate(y = as.numeric(skip & rt < 1000)) %>%
    plot_effect(pre_correct, y) +
    coord_cartesian(xlim=c(NULL), ylim=c(0, 1)) +
    labs(x="Pretest Accuracy", y="Proportion Fast Skip")

p_skip = df %>% 
    plot_effect(pre_correct, 1*skip) +
    labs(x="Pretest Accuracy", y='Proportion Skip') +
    scale_x_continuous(n.breaks=3)

judge_dist = df %>% ggplot(aes(judgement, ..prop.., fill=name)) +
    geom_bar(position="dodge") +
    facet_wrap(~response_type) +
    theme(legend.position="none") +
    pal +
    labs(x="Judgement", y='Proportion')

((acc_rt / judge_rt) | (p_skip + fast_skip) / judge_dist) + plot_annotation(tag_levels = 'A')
(acc_rt | (p_skip + fast_skip)) / (judge_rt + judge_dist) + plot_annotation(tag_levels = 'A')
fig("stopping_full", 10, 6)
```

```{r}
acc_rt = df %>% plot_effect(pre_correct, rt_z) +
    facet_wrap(~response_type) +
    labs(x="Pretest Accuracy", y='Reaction Time') +
    # coord_cartesian(xlim=c(NULL), ylim=c(0, 4000)) +
    scale_x_continuous(n.breaks=3)

judge_rt = df %>% plot_effect(judgement, rt_z) +
    facet_wrap(~response_type) +
    labs(x="Judgement", y='Reaction Time') +
    # coord_cartesian(xlim=c(NULL), ylim=c(0, 4000))
    scale_x_continuous(n.breaks=5)

(acc_rt / judge_rt)
fig("stopping_zscore", 5, 6)
```

```{r}
plot_effect = function(df, x, y) {
    ggplot(df, aes({{x}}, {{y}}, color=name, linetype=name)) +
        stat_summary(fun.data=mean_cl_boot, size=.5) +
        stat_summary(fun=mean, geom="line") +
        theme(legend.position="none") +
        pal
}
```

## Judgement tuning

```{r, fig.width=5, fig.height=3}
make_breaks = function(tail_size) {
    c(0, seq(tail_size, 1-tail_size, length=4), 1)
}

model = read_csv('../model/results/stopping_sim.csv') %>% 
    mutate(response_type = if_else(outcome == 1, "correct", "empty")) %>% 
    # mutate(rt = rt / 1000) %>% 
    mutate(μ_post = μ_post + rnorm(n(), sd=1)) %>% 
    mutate(judgement=cut(μ_post, quantile(μ_post, make_breaks(.35)), labels=F))

trials$name = "Human"
trials$judgement_alt = trials$judgement
model$name = "Model"
df = bind_rows(trials, model) %>% mutate(
    correct = response_type == "correct",
    skip = response_type == "empty",

)

plot_both(judgement, rt) +
    facet_wrap(~response_type) +
    labs(x="Judgement", y='Reaction Time') +
    scale_x_continuous(n.breaks=5)
```

```{r}
df %>% ggplot(aes(judgement, ..prop.., fill=name)) +
    geom_bar(position="dodge") +
    facet_wrap(~response_type) +
    theme(legend.position="none") +
    scale_colour_manual(values=c(
        'Human'='gray10',
        'Model'='#9D6BE0'
    ), aesthetics=c("fill", "colour"), name="") +
    labs(x="Judgement", y='Proportion')
```

## Stopping probability over time

```{r, fig.width=5, fig.height=3}
long = trials %>%
    ungroup() %>% 
    select(wid, word, rt, response_type, pre_correct) %>% 
    mutate(trial_id = row_number()) %>% 
    group_by(trial_id) %>% 
    mutate(n_step = diff(c(0, round(cumsum(rt)/250)))) %>% 
    uncount(n_step, .remove=F) %>% 
    group_by(trial_id) %>% 
    mutate(step = row_number()) %>% 
    mutate(final = as.numeric(step == n_step))

long %>%
    ggplot(aes(step, final, color=pre_correct, group=pre_correct)) +
    stat_summary(fun=mean, geom="line") +
    facet_wrap(~response_type)
```

```{r, fig.width=5, fig.height=3}
trials %>%
    ggplot(aes(rt, color=pre_correct, group=pre_correct)) +
    geom_density() +
    facet_wrap(~response_type) +
    theme(legend.position="top")
```

## FOK model only

```{r, fig.width=5, fig.height=3}
df %>%
    filter(name == "Model") %>% 
    mutate(response_type = if_else(response_type == "correct", "Confidence", "Feeling of Knowing")) %>% 
    group_by(response_type, judgement) %>% 
    summarise(rt = mean(rt)) %>% 
    ggplot(aes(judgement, rt)) +
    geom_line(size=.5) + geom_point(size=2) +
    facet_wrap(~response_type) +
    labs(x="Judgement", y="Reaction Time")
```

## Conditional stopping

```{r}
trials %>% 
    filter(rt > 1140) %>% 
    filter(!(correct & rt < 2000)) %>% 
    mutate(rt2 = as.numeric(rt > 2000)) %>% 
    ggplot(aes(pre_correct, rt2)) +
    stat_summary(fun.data=mean_cl_boot)
```

```{r}
trials %>% 
    filter(!(correct & (rt < 1000))) %>% 
    mutate(y = as.numeric(skip & rt < 1000)) %>%
    # mutate(y = as.numeric(rt > 1000)) %>% 
    ggplot(aes(pre_correct, y)) +
    stat_summary(fun.data=mean_cl_boot, size=.5) +
    stat_summary(fun=mean, geom="line") +
    labs(x="Pretest Accuracy", y="Proportion Skip in First Second")
```

## Joint plots

```{r, fig.width=6, fig.height=3}
fok = trials %>%
    filter(skip) %>% 
    regress(judgement, rt, bins=0, bin_range=1) +
    stat_summary(fun.data=mean_cl_boot, size=.2) +
    xlab("FOK Judgement")

conf = trials %>%
    filter(correct) %>% 
    regress(judgement, rt, bins=0, bin_range=1) +
    stat_summary(fun.data=mean_cl_boot, size=.2) +
    xlab("Confidence Judgement")

(fok + conf) & coord_cartesian(xlim=c(NULL), ylim=c(0, 4000)) #plot
```


```{r, fig.width=6, fig.height=3}
p1 = trials %>% #plot
    filter(correct) %>% 
    regress(pre_correct, rt, bins=0, bin_range=1) +
    stat_summary(fun.data=mean_cl_boot, size=.2) +
    xlab("Pretest Accuracy")

p2 = trials %>% #plot
    filter(skip) %>% 
    regress(pre_correct, rt, bins=0, bin_range=1) +
    stat_summary(fun.data=mean_cl_boot, size=.2) +
    xlab("Pretest Accuracy")

(p1 + p2) & coord_cartesian(xlim=c(NULL), ylim=c(0, 4000))
```

## KDE

```{r}
trials %>% 
    ggplot(aes(rt, color=response_type)) +
    geom_density()
```


## Response types

```{r}
trials %>% 
    count(wid, response_type) %>% 
    pivot_wider(names_from=response_type, values_from=n) %>% 
    replace(is.na(.), 0) %>% 
    select(any_of(c("wid", "correct", "empty", "intrusion", "other", "timeout"))) %>% 
    arrange(correct) %>% kable
```

- there are very few errors (incorrect responses)
- most participants provide empty responses frequently
- some provide empty responses almost all the time
- two out of ten never give an empty response and time out frequently,
  suggesting that they didn't understand the instructions


## Reaction time on success trials

```{r}
trials %>% 
    filter(response_type == "correct") %>% 
    regress(strength, rt_z) #plot
```

```{r}
trials %>% 
    filter(response_type == "correct") %>% 
    regress(judgement, rt_z, bins=0, bin_range=1) +
    stat_summary(fun.data=mean_cl_boot) +
    xlab("Confidence Judgement") #plot
```

- faster correct responses with higher strength, as we've seen before
- slower responses for non-max confidence, no discrimination within
  other confidence values

## Reaction time on empty trials

```{r}
trials %>% 
    filter(response_type == "empty") %>% 
    # group_by(wid) %>% 
    # mutate(strength=zscore(strength)) %>% 
    regress(strength, rt_z, bin_range=.9) #plot
```

```{r}
trials %>% 
    filter(response_type == "empty") %>% 
    group_by(wid) %>% 
    mutate(strength=zscore(strength)) %>% 
    regress(strength, rt_z) #plot
```

```{r}
trials %>% 
    filter(response_type == "empty") %>% 
    regress(judgement, rt_z, bins=0, bin_range=1) +
    stat_summary(fun.data=mean_cl_boot) +
    xlab("Feeling of Knowing Judgement") #plot

```

## Split by participant

```{r,fig.width=WIDTH, fig.height=HEIGHT}
trials %>% 
    # filter(version == "v6.5C") %>% 
    filter(response_type %in% c("correct", "empty")) %>% 
    group_by(wid) %>% 
    mutate(
        rt_z = zscore(rt), 
        mean_empty = mean(response_type == "empty"),
        participant = glue("{100*round(mean_empty, 2)}% empty({wid})")
    ) %>% 
    group_by(wid, response_type) %>% 
    mutate(strength = zscore(strength)) %>% 
    ungroup() %>% 
    mutate(participant=fct_reorder(participant, mean_empty)) %>% 
    ggplot(aes(strength, rt_z, color=mean_empty, group=wid)) + 
    geom_smooth(method="lm", level=0, size=.5) +
    facet_wrap(~response_type) +
    labs(x="Memory Strength", y="Reaction Time (z-scored)")

```

- the effect for correct trials is pretty reliable
- the effect for empty trials only comes out in participants who give empty responses between 
  40% and 75% of the timer


## Metacognitive accuracy

```{r}
trials %>%
    ggplot(aes(judgement, pre_correct, color=judgement_type)) +
    stat_summary(fun.data=mean_cl_boot)
```

```{r}
trials %>%
    ggplot(aes(judgement, ..prop.., fill=factor(pre_correct))) +
    geom_bar(position="dodge") +
    facet_grid(~judgement_type, scales="free")
```

```{r}
trials %>%
    ggplot(aes(pre_correct, ..prop.., fill=judgement, group=judgement)) +
    geom_bar(position="stack") +
    facet_grid(~judgement_type, scales="free")
```

```{r, fig.width=5, fig.height=HEIGHT}
trials %>% 
    ggplot(aes(judgement)) +
    facet_wrap(~judgement_type) +
    geom_bar()
```

```{r, fig.width=5, fig.height=HEIGHT}
trials %>% 
    filter(response_type == "empty") %>%
    count(wid, judgement) %>% 
    pivot_wider(names_from=judgement, values_from=n) %>% 
    replace(is.na(.), 0) %>% 
    arrange(`1`) %>% kable
```

## Any participants with accurate FOK?

Doesn't look like it!

```{r}
fok = trials %>% 
    filter(judgement_type == "fok") %>%
    group_by(wid) %>% 
    # filter(n() >= 5) %>% 
    filter(sum(judgement>1)>=2)

fok %>% 
    group_modify(function(data, grp) {
        lm(strength ~ judgement, data=data) %>% tidy
    }) %>% 
    filter(term == "judgement") %>% 
    arrange(estimate) %>% 
    ungroup() %>% 
    mutate(wid=fct_reorder(wid, estimate)) %>% 
    ggplot(aes(estimate, wid)) +
    geom_point() + geom_vline(xintercept=0)
```

```{r}
trials %>% 
    filter(response_type == "empty") %>% 
    filter(wid == "w7e03059") %>% 
    ggplot(aes(strength, rt_z)) +
    geom_point()
```

```{r}
fok %>% 
    ggplot(aes(judgement, strength, color=wid)) +
    stat_summary(fun=mean, geom="line", size=.3) +
    theme(legend.position="none")
```

```{r}
fok %>% 
    group_by(wid, judgement_type) %>%
    mutate(judgement.z=zscore(judgement)) %>% 
    ggplot(aes(judgement.z, strength, group=wid)) +
    geom_smooth(se=F, method="lm")
```


## Reaction time by response type

```{r,fig.width=WIDTH, fig.height=HEIGHT}
trials %>% 
    ggplot(aes(wid, rt, color=response_type)) +
    geom_quasirandom(size=.1) +
    scale_x_discrete(labels = NULL, breaks = NULL) + 
    labs(x="Participant", y="Reaction Time")
```

## Pretest accuracy

```{r}
pretest %>% 
    group_by(wid) %>%
    summarise(accuracy=mean(correct)) %>% 
    ggplot(aes(accuracy)) + stat_ecdf(geom="point")
```

```{r}
pretest %>% 
    group_by(wid,word) %>%
    summarise(accuracy=mean(correct)) %>% 
    group_by(wid) %>% 
    mutate(acc=mean(accuracy)) %>% 
    ungroup() %>% 
    mutate(wid=fct_reorder(wid, acc)) %>% 
    ggplot(aes(wid,fill=factor(accuracy))) + geom_bar()
    # count(wid, accuracy) %>% 
    # ggplot(aes(accuracy, n, group=wid)) +
    # geom_line(size=.5)
    # pivot_wider(names_from=accuracy, values_from=n)
```

## Strength predicts skipping?

```{r}
trials %>% 
    mutate(skip=response_type=="empty") %>% 
    regress(strength,skip)
```

## Comparison of versions

```{r}
trials %>% 
    filter(response_type == "empty") %>% 
    group_by(wid) %>% 
    # filter(sd(strength) != 0) %>% 
    mutate(strength=zscore(strength)) %>% 
    group_by(version) %>% 
    group_modify(function(data, grp) {
        lmer(rt_z ~ strength + (strength|wid), data=data) %>% tidy
        # lmer(rt_z ~ strength + (strength|wid), data=data) %>% tidy
    }) %>% 
    filter(term == "strength") %>% 
    left_join(
        load_data('participants') %>% count(version)
    )
```









## Trials with backspace

```{r}
all_trials %>% filter(response_type == "empty") %>% 
    filter(type_time > 1) %>% 
    select(base_rt, rt)
```

## Better participants higher FOK?


```{r}
plotcor = function(data) {
    ggplot(data, aes(strength, judgement)) + 
    geom_point() + geom_smooth(method="lm") + facet_wrap(~pilot)
}

reportcor = function(data) {
    data %>% 
        group_by(pilot) %>% 
        group_modify(function(data, grp) {
            data %>% with(cor.test(judgement, strength)) %>% tidy
        }) %>% 
        select(pilot, estimate, p.value) %>%
        kable %>% print
    plotcor(data)
}

trials %>% 
    filter(skip | correct) %>% 
    filter(skip) %>% 
    reportcor
```

```{r}
trials %>% 
    # filter(correct|skip) %>% 
    group_by(wid) %>% 
    mutate(judgement=zscore(judgement), strength=zscore(strength)) %>% 
    filter(skip) %>% 
    reportcor #plot
```


## Early-late split

```{r}
trials %>%
    mutate(phase=if_else(trial_number > 20, "late", "early")) %>% 
    filter(skip) %>% 
    group_by(wid) %>% 
    filter(sd(strength) != 0) %>% 
    mutate(strength=zscore(strength), rt_z = zscore(rt)) %>% 
    group_by(pilot, phase) %>% 
    group_modify(function(data, grp) {
        lmer(rt_z ~ strength + (strength|wid), data=data) %>% tidy
    }) %>% 
    filter(term == "strength") %>% 
    select(pilot, estimate, std.error, p.value)
```


## RT FOK correlation

```{r}
library(ppcor)
select = dplyr::select

trials %>% 
    group_by(wid) %>%
    select(rt, strength, judgement) %>% 
    mutate(across(everything(), zscore)) %>% 
    ungroup() %>% select(-wid) %>% 
    pcor
```

```{r}

X = trials %>% 
    filter(skip) %>% 
    # filter(pilot == "leading instructions") %>% 
    group_by(wid) %>%
    filter(sd(judgement) > 0 & sd(strength) > 0) %>% 
    select(rt, strength, judgement) %>% 
    mutate(across(everything(), zscore)) %>% 
    ungroup() %>% select(-wid)

X %>% pcor %>% with(estimate) %>% kable(digits=3)
X %>% cor %>% kable(digits=3)
```

```{r}
X = trials %>% 
    ungroup() %>% 
    # filter(pilot == "leading instructions") %>% 
    # group_by(wid) %>%
    select(rt, strength, judgement) %>% 
    mutate(across(everything(), zscore))

X %>% pcor %>% with(estimate) %>% kable(digits=3)
X %>% cor %>% kable(digits=3)
```

```{r}
library(mediation)

# strength -> FOK -> RT
# all mediated
med.fit = lm(judgement ~ strength, data=X)
out.fit = lm(rt ~ judgement + strength, data=X)
med = mediate(med.fit, out.fit, treat="strength", mediator="judgement")
summary(med)
```

```{r}
# strength -> rt -> FOK
# mostly direct
med.fit = lm(rt ~ strength, data=X)
out.fit = lm(judgement ~ rt + strength, data=X)
med2 = mediate(med.fit, out.fit, treat="strength", mediator="rt")
summary(med2)
```

```{r}

    %>% 

    mutate(across(rt, st))
    ungroup() %>% 
    filter(skip) %>% 

     %>% pcor


trials %>% 
    ungroup() %>% 
    filter(skip) %>% 
    with(cor.test(rt, strength))

```

```{r}
trials %>% 
    ungroup() %>% 
    filter(skip) %>%
    with(pcor.test(strength, judgement, rt))

trials %>% 
    ungroup() %>% 
    filter(skip) %>%
    with(cor.test(strength, judgement))
```

# Z-scoring

Here is what we get when we z-score RT and judgements within
participants (within correct responses only).

```{r}
trials %>% #plot
    filter(response_type == "correct") %>% 
    group_by(wid) %>% 
    filter(sd(judgement) != 0) %>% 
    mutate(judgement=zscore(judgement), rt_z=zscore(rt)) %>% 
    regress(judgement, rt_z) +
    xlab("Confidence Judgement (z-scored)") +
    coord_cartesian(xlim=c(NULL), ylim=c(-1.5, 1.5))
```

Again, a much larger effect when speed is not
incentivized, although less so when z-scoring:

```{r}
trials %>% #plot
    filter(skip) %>% 
    group_by(wid) %>% 
    filter(sd(judgement) != 0) %>% 
    mutate(judgement=zscore(judgement), rt_z=zscore(rt)) %>% 
    regress(judgement, rt_z) +
    xlab("Feeling of Knowing Judgement (z-scored)") +
    coord_cartesian(xlim=c(NULL), ylim=c(-1.5, 1.5))
```


```{r}
trials %>% 
    filter(response_type == "correct") %>% 
    group_by(wid) %>% 
    mutate(strength_z=zscore(strength), rt_z=zscore(rt)) %>% 
    regress(strength, rt_z) +
    coord_cartesian(ylim=c(-1.5, 1.5))
    #plot
```

## RT KDE

```{r}
trials %>% 
    mutate(response_type = case_when(
        response_type == "correct" ~ "correct",
        response_type == "empty" ~ "empty",
        TRUE ~ "error"
    )) %>% 
    ggplot(aes(rt, color=response_type)) +
    geom_density() +
    facet_wrap(~pilot) +
    coord_cartesian(xlim=c(0,5000), ylim=c(NULL)) +
    scale_colour_manual(values=c(
        "springgreen4",
        "gray", 
        "deeppink3"
    ), aesthetics=c("color"), name="") +
    scale_x_continuous(n.breaks=3)
```


## Strength 


```{r}
trials %>% 
    mutate(pre_logrt = if_else(pre_correct == 0.5))
    ggplot(aes(pre_logrt, 1*correct, color=factor(pre_correct))) +
    stat_summary_bin(fun.data=mean_cl_boot, bins=5) +
    geom_smooth(method="lm")
```

```{r}
glm(skip ~ strength, family=binomial, data=trials) %>% summ(scale=T)
glm(skip ~ pre_correct + pre_logrt, family=binomial, data=trials)
glm(skip ~ pre_correct_z + pre_logrt_z, family=binomial, data=trials)
glm(skip ~ pre_correct + pre_logrt, family=binomial, data=trials)


glmer(skip ~ pre_correct + pre_logrt + (pre_correct + pre_logrt | wid), family=binomial, data=trials) %>% 
    summ(model.fit=T)
lmer(skip ~ pre_correct + pre_logrt + (pre_correct + pre_logrt | wid), data=trials) %>% 
    summ(model.fit=T)

glm(skip ~ pre_correct + pre_logrt, family=binomial, data=trials) %>% 
    summ(model.fit=T)
lm(skip ~ pre_correct + pre_logrt, data=trials) %>% 
    summ(model.fit=T)
```

```{r}
trials %>%
    filter(response_type=="empty") %>%
    group_by(wid, judgement_type) %>%
    mutate(judgement=zscore(judgement)) %>% 
    ggplot(aes(judgement, strength, color=version)) +
    geom_smooth(se=F)
```


Here is the full distribution, as empirical CDFs:
```{r}
trials %>% 
    group_by(pilot, wid) %>% 
    filter(skip) %>% 
    summarise(x=mean(pre_correct==0)) %>% 
    ggplot(aes(x)) + 
    stat_ecdf(geom="point") +
    labs(x="Proportion of skip trials\nwith minimum strength",
         y="Proportion of participants (CDF)") +
    coord_flip() +
    facet_wrap(~pilot)
```
