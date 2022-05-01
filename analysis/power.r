library(simr)
suppressPackageStartupMessages(source("setup.r"))
suppressPackageStartupMessages(source("stats_base.r"))

fixations = load_human("exp2", "fixations") %>%
    filter(response_type == "correct") %>% 
    mutate(
        duration = duration / 1000,
        last_fix = as.numeric(presentation == n_pres),
        fix_first = presentation %% 2,
        fix_stronger = case_when(
            pretest_accuracy_first == pretest_accuracy_second ~ NaN,
            pretest_accuracy_first > pretest_accuracy_second ~ 1*fix_first,
            pretest_accuracy_first < pretest_accuracy_second ~ 1*!fix_first
        )
    )


# %% --------

groups = fixations %>% 
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
    ) %>% 
    mutate(relative = fixated - nonfixated) %>% 
    select(wid, duration, fixated) %>%
    nest_by(wid, .keep=TRUE) %>%
    with(data)

sample_p = function(N, run_model) {
    data = sample(groups, N, replace=T) %>% 
        map(~ mutate(.x, wid=round(1e10 * runif(1)))) %>%
        bind_rows
    tryCatch(run_model(data), error=function(c) NaN)
}

power_analysis = function(N, n_sim, run_model) {
    results = map(N, ~ replicate(n_sim, sample_p(.x, run_model))) %>% unlist
    expand.grid(
        sim_i = 1:n_sim,
        N = N
    ) %>% 
    rowwise() %>% 
    mutate(
        p = sample_p(N, run_model)
    ) %>% ungroup()
}

# %% --------
N = c(300, 350, 400, 450, 500)
n_sim = 300

p1 = power_analysis(N, n_sim, . %>% 
    lmer(duration ~ fixated + (fixated|wid), data=.) %>% 
    summ %>% with(coeftable[2, "p"])
)
# %% --------

p1 %>% 
    group_by(N) %>%
    summarise(power=mean(p<.05, na.rm=T)) %>% kable(digits=3)
# %% --------
p1 %>% 
    group_by(N) %>%
    summarise(power=mean(p<.05, na.rm=T)) %>% 
    ggplot(aes(N, power)) + geom_line() + ylim(0.7, 1)

fig()


