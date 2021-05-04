
# %% ==================== Giving up ====================

trials %>% filter(response_type == "other") %>% with(response)

# for simple v4.0 23 is the number of "give up" responses

23 / nrow(trials)



# %% --------


# %% --------
trials %>% ggplot(aes(log_afc_rt, log_recall_rt)) + geom_smooth()
fig()
# %% ==================== Others ====================


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


    
