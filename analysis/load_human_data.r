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

multi = multi %>% add_strength(
    block == max(block), 
    if_else(correct, -log(rt), -log(15000))
)

multi$name = "Human"

make_fixations = function(df) {
    print("WARNING: USING FULL MULTI IN make_fixations")
    breaks = quantile(abs(multi$rel_strength), c(0, .5, .75, 1),  na.rm = T)
    breaks[4] = Inf
    df %>% 
        ungroup() %>% 
        filter(n_pres >= 1) %>% 
        select(name, wid, rel_strength, presentation_times, n_pres) %>% 
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

long = multi %>% make_fixations

df = multi %>%
    filter(
        response_type == "correct",
        # response_type %in% c("correct", "timeout"),
        # response_type != "intrusion",
    )

keep_acc = multi %>% 
    group_by(wid) %>% 
    summarise(accuracy=mean(correct)) %>% 
    filter(accuracy > 0.5) %>% 
    with(wid)

df = df %>% filter(wid %in% keep_acc)


avg_ptime = long %>% group_by(name,wid) %>% 
    filter(last_fix == 0) %>% 
    # filter(presentation == 1) %>% 
    summarise(duration_mean=mean(duration), duration_sd=sd(duration))

df = df %>% 
    left_join(avg_ptime) %>% 
    mutate(
        first_pres_time_z = (first_pres_time - duration_mean)/duration_sd,
        second_pres_time_z = (second_pres_time - duration_mean)/duration_sd,
        third_pres_time_z = (third_pres_time - duration_mean)/duration_sd,
        fourth_pres_time_z = (fourth_pres_time - duration_mean)/duration_sd,
    )
