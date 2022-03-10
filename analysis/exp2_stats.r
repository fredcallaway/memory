suppressPackageStartupMessages(source("setup.r"))
suppressPackageStartupMessages(source("stats_base.r"))

S = 2.5
MAKE_PDF = TRUE
write_tex = tex_writer("stats/exp2")

# %% ==================== load data ====================

pretest = read_csv('../data/processed/exp2/pretest.csv', col_types = cols())

trials = load_human("exp2", "trials") %>% 
    filter(response_type == "correct") %>% 
    mutate(rel_pretest_accuracy = pretest_accuracy_first - pretest_accuracy_second)

fixations = load_human("exp2", "fixations") %>%
    filter(response_type == "correct") %>% 
    mutate(
        last_fix = as.numeric(presentation == n_pres),
        fix_first = presentation %% 2,
        fix_stronger = case_when(
            pretest_accuracy_first == pretest_accuracy_second ~ NaN,
            pretest_accuracy_first > pretest_accuracy_second ~ 1*fix_first,
            pretest_accuracy_first < pretest_accuracy_second ~ 1*!fix_first
        )
    )
fmt_percent = function(prop) glue("{round(100 * prop)}\\%")


# %% ==================== descriptive stats ====================

# write_tex("N/recruited", length(unique(multi_raw$wid)))
# write_tex("N/recruited", length(unique(multi$wid)) + N_drop_acc)
# write_tex("N/drop_acc", N_drop_acc)
# write_tex("N/analysed", length(unique(multi$wid)))

# %% --------

# simple %>% 
#     group_by(wid) %>% 
#     filter(block == max(block)) %>% 
#     filter(n() == 80) %>% 
#     count(response_type) %>% 
#     mutate(prop=prop.table(n)) %>% 
#     group_by(response_type) %>% 
#     summarise(mean=mean(prop), sd=sd(prop)) %>%
#     rowwise() %>% group_walk(~ with(.x, 
#         write_tex("simple_response_pct/{response_type}", "{100*mean:.1}\\% $\\pm$ {100*sd:.1}\\%")
#     ))

# %% --------
# human %>% 
#     filter(n_pres >= 2) %>% 
#     lmer(prop_first ~ rel_pretest_accuracy * last_pres + (rel_pretest_accuracy * last_pres | wid), data=.) %>% 
#     tidy %>% 
#     filter((term != "(Intercept)") & (are_na(effect) | effect == "fixed")) %>% 
#     rowwise() %>% group_walk(~ with(.x,
#         write_tex("overall_interaction/{term}", regression_tex())
#     ))

# %% --------

trials %>%
    filter(n_pres > 1) %>% 
    mutate(prop_first = total_first / (total_first + total_second)) %>% 
    regress(prop_first ~ rel_pretest_accuracy) %>% 
    write_model("prop_first")

# %% --------

trials %>%
    filter(n_pres > 1) %>% 
    mutate(
        prop_first = total_first / (total_first + total_second),
        odd_n_pres = mod(n_pres, 2),
    ) %>% 
    regress(prop_first ~ rel_pretest_accuracy + odd_n_pres) %>% 
    write_model("prop_first2")

# %% --------

trials %>%
    filter(n_pres > 1) %>% 
    mutate(prop_first = total_first / (total_first + total_second)) %>% 
    regress(prop_first ~ pretest_accuracy_first + pretest_accuracy_second) %>% 
    summ

# %% --------

trials %>% 
    # filter(n_pres > 1) %>% 
    with(mean(choose_first == mod(n_pres, 2))) %>% 
    fmt_percent %>% 
    write_tex("prop_choose_last")

# %% --------
fixations %>% 
    # filter(n_pres > 1) %>% 
    group_by(name, wid, trial_id, is_final=presentation==n_pres) %>% 
    summarise(x=sum(duration)) %>%
    mutate(prop = x / sum(x)) %>%
    filter(is_final) %>% 
    participant_means(prop) %>%
    with(mean(prop)) %>% 
    fmt_percent %>% 
    write_tex("prop_last_duration")


# %% ==================== fixation durations ====================

nonfinal = fixations %>% 
    filter(presentation != n_pres) %>% 
    mutate(
        fixated = case_when(
            mod(presentation, 2) == 1 ~ pretest_accuracy_first,
            mod(presentation, 2) == 0 ~ pretest_accuracy_second,
        ),
        nonfixated = case_when(
            mod(presentation, 2) == 1 ~ pretest_accuracy_second,
            mod(presentation, 2) == 0 ~ pretest_accuracy_first,
        )
    ) %>% mutate(relative = fixated - nonfixated)

# %% --------
nonfinal %>%
    group_by(name, wid) %>% mutate(duration = scale(duration, center=T, scale=F)) %>% 
    participant_means(duration, fixated) %>% 
    group_by(fixated) %>% 
    summarise(mean(duration))

# %% --------
nonfinal %>% 
    regress(duration ~ fixated) %>% 
    summ

# %% --------

nonfinal %>%
    filter(presentation > 1) %>% 
    group_by(name, wid) %>% mutate(duration = scale(duration, center=T, scale=F)) %>% 
    participant_means(duration, nonfixated) %>% 
    group_by(nonfixated) %>% 
    summarise(mean(duration))







