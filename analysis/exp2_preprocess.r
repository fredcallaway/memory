VERSIONS = c('v5.6B')

suppressPackageStartupMessages(source("setup.r"))
source("preprocess_common.r")  # defines all pretest and agg_pretest
write_tex = tex_writer("stats/exp2")

# %% ==================== Load ====================

all_pretest = load_data('simple-recall') %>% 
    filter(!practice) %>% 
    preprocess_recall

all_trials = load_data('multi-recall') %>% 
    filter(!practice) %>% 
    preprocess_recall

# %% ==================== Exclusions ====================

excl = all_trials %>% 
    group_by(wid) %>% 
    filter(n() == 19) %>% 
    summarise(accuracy=mean(response_type=="correct"), n_trial=n()) %>% 
    mutate(incomplete=n_trial != 19, many_error=accuracy < 0.5) %>% 
    mutate(keep = !many_error) %>% 
    filter(!incomplete)  # NOTE: not counting these as "recruited" maybe not ideal

nrow(excl) %>% write_tex("N/recruited")
n_pct(!excl$keep) %>% write_tex("N/excluded")
sum(excl$keep) %>% write_tex("N/final")

keep_wids = excl %>% filter(keep) %>% with(wid)
pretest = all_pretest %>% filter(wid %in% keep_wids)
trials = all_trials %>% filter(wid %in% keep_wids)

# %% ==================== Select and Augment ====================

trials = trials %>% transmute(
    wid, response_type, first_word, second_word,
    raw_rt = rt,
    presentation_times = map(presentation_times, fromJSON),
    first_pres_time = map_dbl(presentation_times, 1, .default=NaN),
    second_pres_time = map_dbl(presentation_times, 2, .default=NaN),
    third_pres_time = map_dbl(presentation_times, 3, .default=NaN),
    last_pres_time = map_dbl(presentation_times, last, default=NaN),
    choose_first = word == first_word,
    n_pres = lengths(presentation_times),
    total_first = map_dbl(presentation_times, ~
        sum(unlist(.x[c(T, F)])),  # .x[c(T,F)]  selects every other entry in the list
        .default=0),
    total_second = replace_na(map_dbl(presentation_times, ~
        sum(unlist(.x[c(F, T)])),
        .default=0), 0),
    rt = total_first + total_second,
    trial_id = row_number(),
)

pretest_performance = summarise_pretest(pretest)
trials = trials %>% 
    left_join(pretest_performance, c("wid", "first_word" = "word")) %>% 
    left_join(pretest_performance, c("wid", "second_word" = "word"), suffix=c("_first", "_second"))

# %% ==================== Unroll fixations ====================

fixations = trials %>% 
    select(wid, presentation_times, n_pres, pretest_accuracy_first, pretest_accuracy_second,
           response_type, choose_first, trial_id ) %>%
    filter(n_pres >= 1) %>% 
    unnest_longer(presentation_times, "duration", indices_to="presentation")

# %% ====================  ====================

trials = trials %>% select(-c(presentation_times, first_word, second_word)) 

trials %>% 
    mutate(drop = response_type != "correct") %T>% 
    with(write_tex(n_pct(drop), "N/error")) %>% 
    filter(!drop) %>% 
    select(-drop) %>% 
    write_csv('../data/processed/exp2/trials.csv')

write_csv(trials, '../data/processed/exp2/trials_witherr.csv')
write_csv(pretest, '../data/processed/exp2/pretest.csv')
write_csv(fixations, '../data/processed/exp2/fixations.csv')
