
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

```{r}
# multi %>% filter(wid == first(participants$wid)) %>% 
#     select(strength_first, strength_second, abs(first_advantage)) %>% 
#     arrange(`abs(first_advantage)`) %>% 
#     pivot_longer(c(strength_first, strength_second)) %>% 
#     ggplot()
# Check javascript score computation
# check_score <- participants %>%
#     select(wid, afc_scores) %>%
#     json_to_columns(afc_scores) %>% 
#     pivot_longer(-wid, names_to="word") %>% 
#     mutate(js_strength = -value) %>% 
#     inner_join(afc_scores)

# # max(check_score$js_score - check_score$score)
# stopifnot(mean(check_score$js_strength - check_score$strength) < .1)
```


    
