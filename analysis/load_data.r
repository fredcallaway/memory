
# %% ==================== Human ====================

VERSIONS = c('v3.4', 'v3.5', 'v3.6')

load_data = function(type) {
    VERSIONS %>% 
    map(~ read_csv(glue('../data/{.x}/{type}.csv'))) %>% 
    bind_rows
}

participants = load_data('participants')

multi = load_data('multi-recall') %>%
    filter(!practice) %>%
    left_join(select(participants, wid, version)) %>% 
    mutate(
        dataset = if_else(version == "v3.6", "new", "old"),
        response_type = factor(response_type, 
            levels=c("correct", "intrusion", "other", "timeout", "empty"),
            # labels=c("Correct", "Intrusion", "Other")
        ),
        correct = response_type == "correct",
        rt = typing_rt,
        presentation_times = map(presentation_times, fromJSON),
        first_pres_time = map_dbl(presentation_times, 1, .default=NaN),
        second_pres_time = map_dbl(presentation_times, 2, .default=NaN),
        third_pres_time = map_dbl(presentation_times, 3, .default=NaN),
        choose_first = word == first_word,
        n_pres = lengths(presentation_times),
        odd_pres = mod(n_pres, 2) == 1,
        last_word = if_else(odd_pres, first_word, second_word),
        last_pres = if_else(n_pres %% 2 == 1, "first", "second"),
        chosen_word = if_else(choose_first, first_word, second_word),
        choose_last_seen = last_word == chosen_word,
        presentation_times_first = map(presentation_times, ~ .x[c(T, F)]),
        presentation_times_second = map(presentation_times, ~ .x[c(F, T)]),
        total_first = map_dbl(presentation_times_first, ~sum(unlist(.x)), .default=0),
        total_second = replace_na(map_dbl(presentation_times_second, ~sum(unlist(.x)), .default=0), 0),
        prop_first = (total_first) / (total_first + total_second)
    ) 


afc = load_data('afc') %>% 
    group_by(wid) %>% 
    mutate(
        trial_num = seq(1:n()),
        log_afc_rt = log(rt),
        presentation = (trial_num-1) %/% 40 + 1
    ) %>% 
    filter(!practice) %>% select(-practice)

afc_scores = afc %>% 
    # filter(block > 1) %>%  ## CRITICAL DECISION
    group_by(wid, word) %>% 
    summarise(raw_strength = -mean(log(rt))) %>%
    group_by(wid) %>% 
    mutate(
        strength = scale(raw_strength),
    )

multi = multi %>% 
    left_join(afc_scores, c("wid", "first_word" = "word")) %>% 
    left_join(afc_scores, c("wid", "second_word" = "word"), suffix=c("_first", "_second")) %>% 
    mutate(
        rel_strength = strength_first - strength_second,
        chosen_strength = if_else(choose_first, strength_first, strength_second),
    ) %>% 
    group_by(wid) %>% 
    mutate(
        trial_num = row_number(),
        typing_rt_z = scale(typing_rt)
    )

# %% ==================== Model ====================

read_sim = function(name, noise_sd=0) {
    read_csv(glue("../model/results/sim_{name}.csv")) %>%  mutate(
        name = !!name,
        strength_first = scale(scale(strength_first) + rnorm(n(), sd=noise_sd)),
        strength_second = scale(scale(strength_second) + rnorm(n(), sd=noise_sd)),
        # strength_first = scale(scale(strength_first) + rnorm(n(), sd=noise_sd)),
        # strength_second = scale(scale(strength_second) + rnorm(n(), sd=noise_sd)),
        response_type = factor(if_else(outcome == -1, "timeout", "correctm"),
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
        choose_first = outcome == 1,
        n_pres = lengths(presentation_times),
        odd_pres = mod(n_pres, 2) == 1,
        rel_strength = strength_first - strength_second,
        # strength_first_bin = cut(strength_first, 5, ordered=T),
        # strength_second_bin = cut(strength_second, 5, ordered=T),
        rt=rt*250,
        rel_present=rel_present*250,
        duration_first=duration_first*250,
        duration_second=duration_second*250,
        first_pres_time=first_pres_time*250,
        second_pres_time=second_pres_time*250,
        third_pres_time=third_pres_time*250,
    )
}



