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
X %>% ggplot(aes(wid, choose_last_seen, color=correct, group=correct)) +
    geom

```


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



    
