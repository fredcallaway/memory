# %% ==================== Setup ====================


VERSIONS = c('v5.6')
DROP_HALF = FALSE
DROP_ACC = TRUE
DROP_ERROR = FALSE
NORMALIZE_FIXATIONS = TRUE
MIXED_DURATIONS = TRUE
OPTIMAL_VERSION = "optimal_prior"
source("setup.r")
source("load_data.r")
if (!MIXED_DURATIONS) info("Using fixed effects models for fixations")

WIDTH = 5.5
HEIGHT = 2.4

# %% --------
full_df = df
full_long = long
df = filter(full_df, name  != "Random Commitment")
long = filter(full_long, name  != "Random Commitment")

# %% ==================== Prob first ====================

df %>%
    filter(response_type == "correct") %>% 
    mutate(choose_first = int(choose_first)) %>% 
    regress(strength_first, choose_first, logistic=TRUE) +
    ylab("Prob Choose First Cue") + ylim(0, 1)

fig("choice", WIDTH, HEIGHT)

# %% ==================== Overall fixation proportion ====================

df %>% 
    filter(n_pres >= 2) %>% 
    regress(rel_strength, prop_first)
fig("overall", WIDTH, HEIGHT)

df %>% 
    filter(n_pres >= 2) %>% 
    regress(rel_strength, last_fix, prop_first)
fig("overall_interaction", WIDTH, HEIGHT)

# %% --------
df %>% 
    filter(n_pres >= 2) %>% 
    group_by(name) %>% 
    group_modify(function(data, grp) {
        m = lm(prop_first ~ strength_first, data=data)
        tibble(beta=m$coefficients[2])
    })


# %% --------
long %>% 
    filter(!last_fix) %>% 
    group_by(trial_id, fix_first) %>% 
    summarise(x=sum(duration)) %>% 
    pivot_wider(names_from=fix_first, values_from=x, names_prefix="fix") %>% 
    replace_na(list(fix0=0)) %>% 
    transmute(prop_first_nonfinal = fix1 / (fix0 + fix1)) %>%
    left_join(df) %>% 
    filter(n_pres >= 2) %>% 
    regress(rel_strength, prop_first_nonfinal)
fig("overall", WIDTH, HEIGHT)


normalized_timestep = function(long) {
    long %>% 
        group_by(trial_id) %>%
        # this somewhat complex method ensures that all trials have exactly 100 steps
        # (this isn't true if you just round duration, as I did initially)
        mutate(percentage_complete = round(100*cumsum(duration / sum(duration)))) %>% 
        mutate(n_step = diff(c(0, percentage_complete))) %>% 
        uncount(n_step) %>% 
        group_by(trial) %>% 
        mutate(normalized_timestep = row_number())
}

# %% --------

long %>% 
    normalized_timestep %>% 
    drop_na(strength_diff) %>% 
    ggplot(aes(normalized_timestep/100, fix_stronger, group = strength_diff, color=strength_diff)) +
    geom_smooth(se=F) + 
    scale_x_continuous(n.breaks=3) +
    ylim(0, 1) +
    facet_grid(~name) +
    labs(x="Normalized Time", y="Probability Fixate\nStronger Cue", color="Strength Difference") +
    geom_hline(yintercept=0.5)

fig("time-course", WIDTH, HEIGHT+.5)

# %% --------
## First fixation
df %>%
    filter(n_pres >= 2) %>% 
    # filter(between(strength_first, -4, 4)) %>% 
    regress(strength_first, first_pres_time, mixed=MIXED_DURATIONS) +
fig("first", WIDTH, HEIGHT)

# %% --------
m = human %>%
    filter(n_pres >= 2) %>% 
    lmer(first_pres_time ~ strength_first + (strength_first|wid), data=.)

# %% --------
tibble(coef(m)$wid) %>% 
    ggplot(aes(strength_first, `(Intercept)`)) + geom_point()
fig()
# %% --------
tibble(coef(m)$wid) %>% 
    ggplot(aes(strength_first)) + geom_histogram() +
    xlab("first fixation slope")
fig("first-fix-slope")


# %% --------
## Second fixation

df %>% 
    filter(n_pres >= 3) %>% 
    mutate(inv_rel_strength = -rel_strength) %>% 
    regress(inv_rel_strength, second_pres_time, mixed=MIXED_DURATIONS) +
    xlab("Second - First Memory Strength")
fig("second", WIDTH, HEIGHT)

# %% --------
## Third fixation

df %>% 
    filter(n_pres >= 4) %>% 
    filter(name  != "Random Commitment") %>% 
    regress(rel_strength, third_pres_time, mixed=MIXED_DURATIONS) + 
    xlab("First - Second Memory Strength")
fig("third", WIDTH, HEIGHT)

# %% --------
## Last-fixation effect

df %>% 
    filter(n_pres >= 2) %>% 
    regress_interaction(rel_strength, last_pres, prop_first)
fig("overall-last", WIDTH, HEIGHT+.5)

# %% --------
## Strength predicts last fixation 

df %>% 
    mutate(last_pres_first = as.numeric(last_pres == "first")) %>% 
    regress(rel_strength, last_pres_first, logistic=TRUE) +
    ylab("Prob First Cue Fixated Last")
fig("last", WIDTH, HEIGHT)

# %% --------
## Last fixations are longer

long %>% 
    filter(name  != "Random Commitment") %>% 
    # filter(name  != "Random") %>% 
    filter(duration <= 5000) %>%
    mutate(type=if_else(last_fix==1, "final", "non-final")) %>% 
    ggplot(aes(duration/1000, fill=type, y = ..width..*..density..)) +
    geom_histogram(position="identity", breaks=seq(0, 5.001, .250), alpha=0.5) +
    facet_grid(~name) +
    theme(legend.position="none") +
    scale_colour_manual(values=c(
        "dodgerblue", "gray"
    ), aesthetics=c("fill", "colour"), name="") +
    labs(x="Fixation Duration (s)", y="Proportion")
    # scale_x_continuous(breaks=seq(-1,5,1))

fig("last-duration", WIDTH, HEIGHT)

# %% --------

long %>% 
    filter(name == "Random") %>% 
    filter(duration < 5000) %>% 
    ggplot(aes(duration, fill=factor(last_fix))) +
    geom_bar(position="stack", alpha=0.5)
fig()
# %% --------
long %>% 
    filter(name == "Human") %>% 
    filter(duration < 5000) %>% 
    ggplot(aes(duration, fill=factor(last_fix))) +
    geom_histogram(position="identity", bins=50, alpha=0.5)
fig()


# %% --------
    facet_grid(~name) + 
    scale_x_discrete(name="Fixation Type", labels=c("Non-final", "Final")) +
    ylab("Duration")

last_diff = long %>% 
    filter(name == "Human") %>% 
    with(tapply(duration, last_fix, mean)) %>% 
    diff

long %>% 
    filter(name == "Human") %>% 
    mutate(is_last = int(last_fix==1)) %>% 
    lmer(duration ~ is_last + (is_last|wid), data=.) %>% 
    summ


