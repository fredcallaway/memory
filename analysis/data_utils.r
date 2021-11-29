read_sim = function(name, noise_sd=0) {
    read_csv(glue("../model/results/sim_{name}.csv"), col_types = cols()) %>%  mutate(
        name = !!name,
        # raw_strength_first = strength_first,
        # raw_strength_second = strength_second,
        strength_first = zscore(zscore(log(strength_first)) + rnorm(n(), sd=noise_sd)),
        strength_second = zscore(zscore(log(strength_second)) + rnorm(n(), sd=noise_sd)),
        response_type = factor(if_else(outcome == -1, "timeout", "correct"),
            levels=c("correct", "intrusion", "other", "timeout", "empty"),
            # labels=c("Correct", "Intrusion", "Other")
        ),
        correct = outcome != -1,
        presentation_times = map(presentation_times, fromJSON),
        rt = duration_first + duration_second,
        prop_first = duration_first / rt,
        rel_present = duration_first - duration_second,
        first_pres_time = map_dbl(presentation_times, 1, .default=NaN),
        second_pres_time = map_dbl(presentation_times, 2, .default=NaN),
        third_pres_time = map_dbl(presentation_times, 3, .default=NaN),
        fourth_pres_time = map_dbl(presentation_times, 4, .default=NaN),
        choose_first = outcome == 1,
        n_pres = lengths(presentation_times),
        odd_pres = mod(n_pres, 2) == 1,
        rel_strength = strength_first - strength_second,
        chosen_strength = if_else(choose_first, strength_first, strength_second),
        # strength_first_bin = cut(strength_first, 5, ordered=T),
        # strength_second_bin = cut(strength_second, 5, ordered=T),
        # rt=rt*250,
        # rel_present=rel_present*250,
        # duration_first=duration_first*250,
        # duration_second=duration_second*250,
        # first_pres_time=first_pres_time*250,
        # second_pres_time=second_pres_time*250,
        # third_pres_time=third_pres_time*250,
        # log_strength_first = zscore(zscore(log(raw_strength_first)) + rnorm(n(), sd=noise_sd)),
        # log_strength_second = zscore(zscore(log(raw_strength_second)) + rnorm(n(), sd=noise_sd)),
        # log_rel_strength = log_strength_first - log_strength_second,
    )
}

make_fixations = function(df) {
    breaks = quantile(abs(multi$rel_strength), c(0, .5, .75, 1),  na.rm = T)
    breaks[4] = Inf
    df %>% 
        ungroup() %>% 
        filter(n_pres >= 1) %>% 
        select(name, wid, rel_strength, presentation_times, n_pres, trial_id) %>% 
        mutate(
            strength_diff = cut(abs(rel_strength), breaks,
                                labels=c("small", "moderate", "large"),
                                ordered=T)
        ) %>% 
        unnest_longer(presentation_times, "duration", indices_to="presentation") %>% 
        mutate(
            last_fix = as.numeric(presentation == n_pres),
            fix_first = presentation %% 2,
            fix_stronger = as.numeric(fix_first == (rel_strength > 0)),
            # duration = if_else(name == "Human", as.integer(duration), as.integer(duration * 250)),
        )
}