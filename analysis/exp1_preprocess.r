# VERSIONS = c('v6.5', 'v6.5B', 'v6.5C', 'v6.6', 'v6.7', 'v6.8')
VERSIONS = c('v6.5E')

suppressPackageStartupMessages(source("setup.r"))
source("preprocess_common.r")  # defines all pretest and agg_pretest
write_tex = tex_writer("stats/exp1")

# %% ==================== Load  ====================

all_pretest = load_data('simple-recall') %>% 
    filter(!practice) %>% 
    preprocess_recall

all_trials = load_data('simple-recall-penalized') %>% 
    filter(!practice) %>% 
    preprocess_recall %>% 
    select(-judgement_type)

# %% ==================== Exclusions ====================

excl = all_trials %>% 
    group_by(wid) %>%
    summarise(skip_rate=mean(response_type == "empty")) %>% 
    mutate(keep=skip_rate < .9)

nrow(excl) %>% write_tex("N/recruited")
sum(!excl$keep) %>% write_tex("N/excluded")
sum(excl$keep) %>% write_tex("N/final")

keep_wids = excl %>% filter(keep) %>% with(wid)
pretest = all_pretest %>% filter(wid %in% keep_wids)
trials = all_trials %>% filter(wid %in% keep_wids)

trials = trials %>% 
    mutate(drop = response_type %nin% c("correct","empty")) %T>% 
    with(write_tex(n_pct(drop), "N/error")) %>% 
    filter(!drop) %>% 
    mutate(drop = is.na(rt)) %T>% 
    with(write_tex(sum(drop), "N/short")) %>% 
    filter(!drop)



# %% ==================== Select and augment ====================

trials = trials %>%
    left_join(summarise_pretest(pretest)) %>% 
    select(wid, response_type, rt, judgement, pretest_accuracy)

pretest = pretest %>% select(wid, block, word, response_type, rt)

# %% ==================== Save ====================

write_csv(pretest, '../data/processed/exp1/pretest.csv')
write_csv(trials, '../data/processed/exp1/trials.csv')
write_csv(excl, '../data/processed/exp1/participants.csv')


# judgement_breaks = function(tail_size) {
#     c(0, seq(tail_size, 1-tail_size, length=4), 1)
# }

# read_sim  = . %>% 
#     read_csv() %>% 
#     mutate(response_type = if_else(outcome == 1, "correct", "empty")) %>% 
#     mutate(μ_post = μ_post + rnorm(n(), sd=1)) %>% 
#     mutate(judgement=cut(μ_post, quantile(μ_post, judgement_breaks(.35)), labels=F))

# df = bind_rows(
#     read_sim('../model/results/stopping_sim_rand.csv') %>% mutate(name="Random"),
#     read_sim('../model/results/stopping_sim.csv') %>% mutate(name="Optimal"),
#     trials %>% mutate(name = "Human")
# ) %>% mutate(
#     correct = response_type == "correct",
#     skip = response_type == "empty",

# )