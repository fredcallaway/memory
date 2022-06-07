library(simr)
suppressPackageStartupMessages(source("setup.r"))
suppressPackageStartupMessages(source("stats_base.r"))
source("power.r")

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

N = c(50)
n_sim = 10

p1 = power_analysis(groups, N, n_sim, . %>% 
    lm(rt ~ pretest_accuracy, data=.) %>% 
    summ %>% with(coeftable[2, "p"])
)


# %% --------

N = c(300, 350, 400, 450, 500)
n_sim = 500

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

# %% --------

p2 = power_analysis(N, n_sim, . %>% 
    lm(rt ~ pretest_accuracy) %>% 
    summ %>% with(coeftable[2, "p"])
)

p2 %>% 
    group_by(N) %>%
    summarise(power=mean(p<.05, na.rm=T))

# %% --------

sample_stat = function(groups, N, statistic) {
    data = sample(groups, N, replace=T) %>% 
        map(~ mutate(.x, wid=round(1e10 * runif(1)))) %>%
        bind_rows
    tryCatch(statistic(data), error=function(c) NaN)
}

power_analysis = function(groups, N, n_sim, statistic) {
    results = map(N, ~ replicate(n_sim, sample_stat(groups, .x, statistic))) %>% unlist
    expand.grid(
        sim_i = 1:n_sim,
        N = N
    ) %>% 
    rowwise() %>% 
    mutate(
        stat = sample_stat(groups, N, statistic)
    ) %>% ungroup()
}

groups = trials %>% 
    filter(skip) %>% 
    group_by(wid, pretest_accuracy) %>% 
    summarise(y=median(rt))  %>% 
    ungroup() %>% 
    nest_by(wid, .keep=TRUE) %>%
    with(data)


statistic = . %>% 
    group_by(pretest_accuracy) %>% 
    summarise(mean_cl_boot(y)) %>% 
    select(-y) %>% 
    pivot_wider(names_from=pretest_accuracy, values_from=c(ymin,ymax)) %>% 
    with(ymin_1 - ymax_0)

N = c(300, 350, 400, 450, 500)
n_sim = 100

ci_pwr = power_analysis(groups, N, n_sim, statistic)
# %% --------

ci_pwr %>% 
    group_by(N) %>% 
    summarise(mean(stat > 0))

ci_pwr %>% 
    ggplot(aes(N, stat)) +
    geom_point()

fig()

