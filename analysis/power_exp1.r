library(simr)
suppressPackageStartupMessages(source("setup.r"))
suppressPackageStartupMessages(source("stats_base.r"))

# %% --------
trials = load_human("exp1", "trials") %>% 
    mutate(
        rt = rt / 1000,
        skip=response_type=="empty", correct=response_type=="correct",
        response_type = recode_factor(response_type, "correct" = "Recalled", "empty" = "Skipped")
    )

# %% --------

groups = trials %>%
    filter(skip) %>% 
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

N = c(200, 250, 300, 350, 400)
n_sim = 100

p1 = power_analysis(N, n_sim, . %>% 
    regress(rt ~ pretest_accuracy) %>% 
    summ %>% with(coeftable[2, "p"])
)
# %% --------
p1 %>% 
    group_by(N) %>%
    summarise(power=mean(p<.05, na.rm=T)) %>% 

    ggplot(aes(N, power)) + geom_line()

fig()

