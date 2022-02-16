VERSIONS = c('v5.6')
DROP_ACC = TRUE
DROP_ERROR = TRUE

suppressPackageStartupMessages(source("setup.r"))

load_data = function(type) {
    VERSIONS %>% 
    map(~ read_csv(glue('../data/{.x}/{type}.csv'), col_types = cols())) %>% 
    bind_rows
}

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
        total_time = rt + type_time,
        correct = response_type == "correct",
        base_rt = rt,
        rt = replace_na(typing_rt, 15000),
        logrt_z = zscore(log(rt))
    ) %>% 
    rowwise() %>% mutate(rt = min(rt, 15000)) %>% ungroup()

pretest_performance = simple_raw %>% 
    # filter(trial_num > 20) %>% 
    filter(block == max(block)) %>% 
    rename(pre_correct = correct) %>% 
    mutate(pre_logrt = if_else(pre_correct, log(rt), NaN)) %>% 
    mutate(raw_strength = -if_else(pre_correct, log(rt), log(15000))) %>% 
    group_by(wid, word) %>% 
    summarise(across(c(pre_correct, pre_logrt, raw_strength), mean, na.rm=T)) %>%
    group_by(wid) %>%
    mutate(across(c(pre_correct, pre_logrt, raw_strength), zscore, .names="{.col}_z")) %>% 
    rename(strength = raw_strength_z)


multi_raw = load_data('multi-recall')

multi = multi_raw %>% 
    filter(!practice) %>%
    group_by(wid) %>% 
    # filter(n() == 19) %>% 
    left_join(select(participants, wid, version)) %>% 
    ensure_column(c("primed", "primed_word")) %>% 
    mutate(
        response_type = factor(response_type, 
            levels=c("correct", "intrusion", "other", "timeout", "empty"),
            # labels=c("Correct", "Intrusion", "Other")
        ),
        rt = typing_rt,
        presentation_times = map(presentation_times, fromJSON),
        first_pres_time = map_dbl(presentation_times, 1, .default=NaN),
        second_pres_time = map_dbl(presentation_times, 2, .default=NaN),
        third_pres_time = map_dbl(presentation_times, 3, .default=NaN),
        choose_first = word == first_word,
        n_pres = lengths(presentation_times),
        total_first = map_dbl(presentation_times, ~
            sum(unlist(.x[c(T, F)])),  # .x[c(T,F)]  selects every other entry in the list
            .default=0),
        total_second = replace_na(map_dbl(presentation_times, ~
            sum(unlist(.x[c(F, T)])),
            .default=0), 0),
        trial_num = row_number(),
    )

#  [1] 0.7521722 0.2336804 0.1455062 0.2888899 0.8416949 0.8710598 0.3264926 0.6721940 0.8277123 0.2933744          


multi = multi %>%
    select(-contains("strength")) %>% 
    left_join(pretest_performance, c("wid", "first_word" = "word")) %>% 
    left_join(pretest_performance, c("wid", "second_word" = "word"), suffix=c("_first", "_second")) %>% 
    mutate(
        name = "Human",
        rel_pre_correct = pre_correct_first - pre_correct_second,
        rel_pre_correct_z = (pre_correct_z_first - pre_correct_z_second) / sqrt(2),
        rel_pre_logrt = pre_logrt_first - pre_logrt_second,
        rel_pre_logrt_z = (pre_logrt_z_first - pre_logrt_z_second) / sqrt(2),
        rel_strength = (strength_first - strength_second) / sqrt(2),  # keep it standardized
        chosen_strength = if_else(choose_first, strength_first, strength_second),
        chosen_pre_correct = if_else(choose_first, pre_correct_first, pre_correct_second),
        chosen_pre_logrt_z = if_else(choose_first, pre_logrt_z_first, pre_logrt_z_second),
    ) %>% ungroup() %>% mutate(trial_id = row_number())

if (DROP_ACC) {
    keep_acc = multi %>% 
        group_by(wid) %>% 
        summarise(accuracy=mean(response_type=="correct")) %>% 
        filter(accuracy > 0.5) %>% 
        with(wid)
    N_total = multi %>% with(length(unique(wid)))
    N_drop_acc = N_total - length(keep_acc)
    multi = multi %>% filter(wid %in% keep_acc)
    simple = simple %>% filter(wid %in% keep_acc)
    print(glue('Dropping {N_drop_acc} participants with less than 50% accuracy in critical trials'))
}

if (DROP_ERROR) {
    print("Dropping error trials")
    multi = multi %>% 
        filter(
            response_type == "correct",
            # response_type %in% c("correct", "timeout"),
            # response_type != "intrusion",
        )
} else {
    print("_Including_ error trials")
}

unroll_fixations = function(df) {
    df %>% 
        ungroup() %>% 
        filter(n_pres >= 1) %>% 
        select(wid, trial_id, presentation_times, n_pres, rel_pre_correct) %>% 
        unnest_longer(presentation_times, "duration", indices_to="presentation") %>% 
        mutate(
            last_fix = as.numeric(presentation == n_pres),
            fix_first = presentation %% 2,
            fix_stronger = as.numeric(fix_first == (rel_pre_correct > 0)),

            # duration = if_else(name == "Human", as.integer(duration), as.integer(duration * 250)),
        )
}

fixations = unroll_fixations(multi)
multi %>% select(-presentation_times) %>% write_csv('../data/processed/exp2/trials.csv')
simple %>% write_csv('../data/processed/exp2/pretest.csv')
fixations %>% write_csv('../data/processed/exp2/fixations.csv')


