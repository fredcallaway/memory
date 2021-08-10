library(simr)



# %% ==================== Aug 4 ====================
human %>% 
    filter(n_pres >= 2) %>% 
    lmer(first_pres_time ~ strength_first + (strength_first|wid), data=.) %>% 
    summ()

# %% --------

model = human %>% 
    filter(n_pres >= 2) %>% 
    lm(first_pres_time ~ strength_first, data=.)


# %% --------
model = human %>% 
    filter(n_pres >= 3) %>% 
    lm(second_pres_time ~ rel_strength, data=.)


powerSim(extend(model, along="wid", n=150), nsim=100)
powerSim(model, nsim=100)

# %% --------

groups = human %>% nest_by(wid, .keep=TRUE) %>% with(data)

sample_p = function(N) {
    bind_rows(sample(groups, N, replace=T)) %>% 
        filter(n_pres >= 2) %>% 
        lm(first_pres_time ~ strength_first, data=.) %>% 
        tidy %>% 
        with(p.value[2])
}

n_sim = 300
N = seq(100, 500, 50)
results = map(N, ~ replicate(n_sim, sample_p(.x)))
power = unlist(map(results, ~ mean(.x < .05)))
fixed_pwr = tibble(N, power)
fixed_pwr
# %% --------

groups = human %>% nest_by(wid, .keep=TRUE) %>% with(data)

sample_p = function(N) {
    data = sample(groups, N, replace=T) %>% 
        map(~ mutate(.x, wid=round(1e10 * runif(1)))) %>%
        bind_rows
    model = data %>% 
        filter(n_pres >= 2) %>% 
        lmer(first_pres_time ~ strength_first + (strength_first|wid), data=.)
    summ(model)$coeftable["strength_first", "p"]
}

results = map(N, ~ replicate(n_sim, sample_p(.x)))
power = unlist(map(results, ~ mean(.x < .05)))
mixed_pwr = tibble(N, power)

# %% --------

mixed_pwr %>% ggplot(aes(N, power)) + geom_line() + geom_hline(yintercept=0.8) + 
    ggtitle("First Fixation Duration") + expand_limits(y=1)
fig()

# %% --------
sample_p = function(data, N) {
    independence_test(route_cost ~ feedback == "meta",
                       data=resample(data, N),
                       alternative="greater") %>% pvalue
}








# %% ==================== Previous ====================



model = human %>% 
    filter(response_type == "correct") %>% 
    filter(n_pres > 0) %>% 
    mutate(
        last_pres_time = map_dbl(presentation_times, last),
        last_rel_strength = if_else(last_pres == "first", rel_strength, 1 - rel_strength),
        last_strength = if_else(last_pres == "first", strength_first, strength_second)
    ) %>% 
    lmer(last_pres_time ~ last_strength + (1|wid), data=.)
# %% --------

model = human %>% 
    filter(n_pres >= 4) %>% 
    lmer(third_pres_time ~ rel_strength + (rel_strength|wid), data=.)

human %>% 
    group_by(wid) %>% 
    summarise(n=sum(n_pres > 3)) %>% 
    ggplot(aes(n)) + geom_bar() + labs(x="Number of trials with 4 or more fixations", y="Number of participants")

# %% --------

avg_ptime = long %>% group_by(name,wid) %>% 
    filter(last_fix == 0) %>% 
    summarise(mean=mean(duration), sd=sd(duration))

avg_ptime %>% 
    ggplot(aes(fct_reorder(wid, mean), mean, ymin=mean-sd, ymax=mean+sd)) +
    geom_pointrange()

# %% --------

human %>% 
    filter(n_pres >= 2) %>% 
    left_join(avg_ptime) %>% 
    mutate(y=(first_pres_time-mean)/sd) %>% 
    lm(y ~ strength_first, data=.) %>% 
    summ(scale=T, transform.response = TRUE)

# human %>% 
#     filter(n_pres >= 2) %>% 
#     lmer(first_pres_time ~ strength_first + (1|wid), data=.) %>% 
#     summ(scale=T, transform.response = TRUE)

human %>% 
    filter(n_pres >= 2) %>% 
    lmer(first_pres_time ~ strength_first + (strength_first|wid), data=.) %>% 
    summ(scale=T, transform.response = TRUE)

# %% --------

human %>% 
    filter(n_pres >= 3) %>% 
    left_join(avg_ptime) %>% 
    mutate(y=(second_pres_time-mean)/sd) %>% 
    lm(y ~ rel_strength, data=.) %>% 
    summ(scale=T, transform.response = T)

human %>% 
    filter(n_pres >= 3) %>% 
    lmer(second_pres_time ~ rel_strength + (1|wid), data=.) %>% 
    summ(scale=T, transform.response = TRUE)

human %>% 
    filter(n_pres >= 3) %>% 
    lmer(second_pres_time ~ rel_strength + (rel_strength|wid), data=.) %>% 
    summ(scale=T, transform.response = TRUE)

# %% --------

human %>% 
    filter(n_pres >= 4) %>% 
    left_join(avg_ptime) %>% 
    mutate(y=(third_pres_time-mean)/sd) %>% 
    lm(y ~ rel_strength, data=.) %>% 
    summ(scale=T, transform.response = TRUE)


# human %>% 
#     filter(n_pres >= 4) %>% 
#     lmer(third_pres_time ~ rel_strength + (1|wid), data=.) %>% 
#     summ(scale=T, transform.response = TRUE)

human %>% 
    filter(n_pres >= 4) %>% 
    # filter(response_type %in% c("correct", "timeout")) %>% 
    lmer(third_pres_time ~ rel_strength + (rel_strength|wid), data=.) %>% 
    summ(scale=T, transform.response = TRUE)

# %% --------

human %>% 
    filter(n_pres >= 3) %>% 
    left_join(avg_ptime) %>% 
    mutate(y=(first_pres_time-mean)/sd) %>% 
    ggplot(aes(strength_first, y)) +
    geom_smooth(method="lm") +
    # geom_smooth(aes(group=wid), method="lm", se=F, size=0.2, color="black") +
    stat_summary_bin(fun.data=mean_cl_boot, bins=5) +
    labs(x="First Cue Memory Strength", y="Normalized First Fixation Time")

fig()
# %% --------

human %>% 
    filter(n_pres >= 2) %>% 
    regress(strength_first, first_pres_time) # plot
fig()
# %% --------

human %>% 
    filter(n_pres >= 2) %>% 
    lmer(first_pres_time ~ strength_first + (strength_first|wid), data=.) %>% 
    summ


# %% --------


powerSim(model, nsim=100)

pc = powerCurve(model, nsim=5, within="wid")


human %>% 
    filter(n_pres >= 4) %>% 
    with(table(wid))



# %% --------
powerCurve(model)

# %% --------
model = lmer(second_pres_time ~ rel_strength + (rel_strength|wid), data=human)
model %>% summ


doTest(model, fixed("rel_strength"))
powerSim(model, nsim=100)
# %% --------
model = lm(second_pres_time ~ -rel_strength, data=human)
powerSim(model, nsim=100)

# %% --------
data = read_csv("SIMR_RW_example.csv")
model1 <- glmer(Pr_rate ~ Day + (1|Doctor), family="poisson", data=data)
powerSim(model1, nsim=100)

# %% --------

