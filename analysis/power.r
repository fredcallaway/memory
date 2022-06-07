
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

