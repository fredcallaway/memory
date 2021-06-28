
# %% ==================== Human ====================

VERSIONS = c('v5.0')

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



simple = load_data('simple-recall') %>% 
    filter(!practice) %>% 
    group_by(wid) %>% 
    filter(n() == 59) %>% 
    mutate(
        trial_num = seq(2,60),
        round = if_else(trial_num <= 20, 1, 2),
        typing_rt_z = zscore(typing_rt)
    ) %>% 
    mutate(
        response_type = factor(response_type, 
            levels=c("correct", "intrusion", "other", "timeout", "empty"),
            # labels=c("Correct", "Intrusion", "Other")
        ),
        word_type = factor(word_type, 
            levels=c("low", "high"), labels=c("Low", "High")),
        total_time = rt + type_time,
        correct = response_type == "correct"
    ) %>% mutate(
        base_rt = rt,
        rt = replace_na(typing_rt, 15000),
        logrtz = zscore(log(rt))
    )


add_strength = function(multi, filt, strength) {
    strengths = simple %>% 
        filter({{ filt }}) %>% 
        mutate(raw_strength={{ strength }}) %>% 
        group_by(wid, word) %>% 
        summarise(raw_strength = mean(raw_strength)) %>%
        group_by(wid) %>% 
        mutate(
            strength = zscore(raw_strength),
        )
    multi %>% 
        select(-contains("strength")) %>% 
        left_join(strengths, c("wid", "first_word" = "word")) %>% 
        left_join(strengths, c("wid", "second_word" = "word"), suffix=c("_first", "_second")) %>% 
        mutate(
            rel_strength = strength_first - strength_second,
            chosen_strength = if_else(choose_first, strength_first, strength_second),
        )
}

multi = multi %>% add_strength(round > 1, 2 * correct -log(rt))

# %% ==================== Model ====================

read_sim = function(name, noise_sd=0) {
    read_csv(glue("../model/results/sim_{name}.csv")) %>%  mutate(
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
    df %>% 
        ungroup() %>% 
        filter(n_pres >= 1) %>% 
        select(name, rel_strength, presentation_times, n_pres) %>% 
        mutate(
            trial = row_number(),
            strength_diff = cut(abs(rel_strength), 
                                c(0, 0.35, 1.25, 10),
                                # quantile(abs(rel_strength), c(0, 0.2, 1.),  na.rm = T),
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


