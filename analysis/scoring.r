

multi %>% 
    add_strength(block > 2, 2 * correct -log(rt)) %>% 
    filter(response_type == "correct") %>% 
    with(lmer(rt ~ chosen_strength + (chosen_strength|wid), data=.)) %>% summ

# %% --------
multi %>% 
    add_strength(block > 2, 2 * correct -log(rt)) %>% 
    # filter(response_type == "correct") %>% 
    with(lmer(choose_first ~ strength_first + (strength_first|wid), data=.)) %>% summ

# %% --------

multi %>% 
    add_strength(block > 2, 2 * correct -log(typing_rt)) %>% 
    filter(n_pres >= 3) %>% 
    with(lmer(second_pres_time ~ rel_strength + (rel_strength|wid), data=.)) %>% summ