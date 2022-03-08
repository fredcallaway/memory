VERSIONS = c('v5.6')
DROP_ACC = TRUE
DROP_ERROR = TRUE

suppressPackageStartupMessages(source("setup.r"))
source("preprocess_common.r")  # defines all pretest and agg_pretest

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
    summarise(accuracy=mean(response_type=="correct")) %>% 
    mutate(keep = accuracy > 0.5)
keep_wids = excl %>% filter(keep) %>% with(wid)
pretest = all_pretest %>% filter(wid %in% keep_wids)
trials = all_trials %>% filter(wid %in% keep_wids)


# %% ==================== Select and Augment ====================

trials = trials %>% transmute(
    wid, response_type, rt, first_word, second_word,
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
)

pretest_performance = summarise_pretest(pretest)
trials = trials %>% 
    left_join(pretest_performance, c("wid", "first_word" = "word")) %>% 
    left_join(pretest_performance, c("wid", "second_word" = "word"), suffix=c("_first", "_second"))

# %% ==================== Unroll fixations ====================

fixations = trials %>% 
    transmute(
        wid, presentation_times, n_pres, pretest_accuracy_first, pretest_accuracy_second, response_type, choose_first,
        trial_id = row_number(),
    ) %>% 
    filter(n_pres >= 1) %>% 
    unnest_longer(presentation_times, "duration", indices_to="presentation")

# %% ====================  ====================

trials %>% 
    select(-c(presentation_times, first_word, second_word)) %>% 
    write_csv('../data/processed/exp2/trials.csv')

write_csv(pretest, '../data/processed/exp2/pretest.csv')
write_csv(fixations, '../data/processed/exp2/fixations.csv')
