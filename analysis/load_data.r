# %% ==================== Human ====================

info = function(txt) {
    cat(glue("- {txt}\n\n"))
}

load_data = function(type) {
    VERSIONS %>% 
    map(~ read_csv(glue('../data/{.x}/{type}.csv'), col_types = cols())) %>% 
    bind_rows
}

participants = load_data('participants')
multi_raw = load_data('multi-recall')

multi = multi_raw %>% 
    filter(!practice) %>%
    group_by(wid) %>% 
    filter(n() == 19) %>% 
    left_join(select(participants, wid, version)) %>% 
    ensure_column(c("primed", "primed_word")) %>% 
    mutate(
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
        fourth_pres_time = map_dbl(presentation_times, 4, .default=NaN),
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
        prop_first = (total_first) / (total_first + total_second),
        trial_num = row_number(),
        first_primed = primed_word == first_word,
    )


simple_raw = simple = load_data('simple-recall') %>% 
    group_by(wid) %>%
    mutate(trial_num = row_number()) %>%
    filter(!practice) %>% 
    # filter(n() == 79) %>% 
    mutate(
        typing_rt_z = zscore(typing_rt),
        response_type = factor(response_type, 
            levels=c("correct", "intrusion", "other", "timeout", "empty"),
            # labels=c("Correct", "Intrusion", "Other")
        ),
        # word_type = factor(word_type, levels=c("low", "high"), labels=c("Low", "High")),
        total_time = rt + type_time,
        correct = response_type == "correct",
        base_rt = rt,
        rt = replace_na(typing_rt, 15000),
        logrtz = zscore(log(rt))
    ) %>% 
    rowwise() %>% mutate(rt = min(rt, 15000)) %>% ungroup()

compute_strength = function(filt, strength) {
    simple %>% 
        filter({{ filt }}) %>% 
        mutate(raw_strength={{ strength }}) %>% 
        group_by(wid, word) %>% 
        summarise(raw_strength = mean(raw_strength)) %>%
        group_by(wid) %>% 
        mutate(
            strength = zscore(raw_strength),
        )
}

add_strength = function(multi, filt, strength) {
    strengths = compute_strength({{filt}}, {{strength}})
    multi %>% 
        select(-contains("strength")) %>% 
        left_join(strengths, c("wid", "first_word" = "word")) %>% 
        left_join(strengths, c("wid", "second_word" = "word"), suffix=c("_first", "_second")) %>% 
        mutate(
            rel_strength = (strength_first - strength_second) / sqrt(2),  # keep it standardized
            chosen_strength = if_else(choose_first, strength_first, strength_second),
        )
}

simple %>% 
    filter(block == max(block)) %>% 
    mutate(raw_strength = 5 * correct - log(rt)) %>% 
    group_by(wid, word) %>% 
    summarise(raw_strength = mean(raw_strength))


multi = multi %>% add_strength(block == max(block), 5 * correct - log(rt))

# %% ==================== Model ====================

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
        rel_strength = (strength_first - strength_second) / sqrt(2),
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
    print("WARNING: USING FULL MULTI IN make_fixations")
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

if (DROP_ACC) {
    keep_acc = multi %>% 
        group_by(wid) %>% 
        summarise(accuracy=mean(correct)) %>% 
        filter(accuracy > 0.5) %>% 
        with(wid)
    N_total = multi %>% with(length(unique(wid)))
    N_drop_acc = N_total - length(keep_acc)
    multi = multi %>% filter(wid %in% keep_acc)
    simple = simple %>% filter(wid %in% keep_acc)
    info(glue('Dropping {N_drop_acc} participants with less than 50% accuracy in critical trials'))
}
if (DROP_HALF) {
    info("Dropping the first half of critical trials")
    multi = multi %>% filter(trial_num > 10)
}

# %% --------
df = raw_df = bind_rows(
    read_sim("new_optimal", noise_sd=1),
    # read_sim("empirical_commitment", noise_sd=1),
    # read_sim("rand_fit", noise_sd=0),
    multi %>% mutate(name = "human", wid = factor(wid)),
    read_sim("new_random", noise_sd=1),
    # read_sim("empirical", noise_sd=1),
) %>% mutate(
    name = recode_factor(name, .ordered=T,
        "sanity_optimal" = "Optimal",
        "optimal" = "Optimal",
        "optimal_prior" = "Optimal",
        "new_optimal" = "Optimal",
        "human" = "Human",
        "sanity_rand" = "Random",
        "empirical" = "Random",
        "new_random" = "Random",
        "empirical_commitment" = "Random Commitment",
        "rand_fit" = "Random Fit"
        # "rand_gamma" = "Random",
    ),
    last_pres = if_else(n_pres %% 2 == 1, "first", "second"),
    trial_id = row_number(),

)

if (DROP_ERROR) {
    info("Dropping error trials")
    df = raw_df %>% 
        filter(
            response_type == "correct",
            # response_type %in% c("correct", "timeout"),
            # response_type != "intrusion",
        )
} else {
    info("_Including_ error trials")
}

long = df %>% 
    # group_by(name) %>%
    # slice_sample(n=1000) %>% 
    make_fixations

if (NORMALIZE_FIXATIONS) {
    info("Z-scoring fixation durations")
    avg_ptime = long %>% group_by(name,wid) %>% 
        filter(last_fix == 0) %>% 
        summarise(duration_mean=mean(duration), duration_sd=sd(duration))
    df = df %>% 
        left_join(avg_ptime) %>% 
        mutate(
            first_pres_time_raw = first_pres_time,
            second_pres_time_raw = second_pres_time,
            third_pres_time_raw = third_pres_time,
            fourth_pres_time_raw = fourth_pres_time,
            first_pres_time = (first_pres_time_raw - duration_mean)/duration_sd,
            second_pres_time = (second_pres_time_raw - duration_mean)/duration_sd,
            third_pres_time = (third_pres_time_raw - duration_mean)/duration_sd,
            fourth_pres_time = (fourth_pres_time_raw - duration_mean)/duration_sd,
        )
}


optimal = filter(df, name == "Optimal")
random = filter(df, name == "Random")
human = filter(df, name == "Human")
info(glue("{length(unique(human$wid))} participants and {nrow(human)} trials in the analysis"))