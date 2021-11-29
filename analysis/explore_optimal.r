
VERSIONS = c('v5.6')
DROP_HALF = FALSE
DROP_ACC = TRUE
DROP_ERROR = TRUE
NORMALIZE_FIXATIONS = TRUE
MIXED_DURATIONS = TRUE
source("setup.r")
source("data_utils.r")
source("load_human_data.r")

raw_df = read_sim("many_optimal")
# %% --------

df = raw_df %>% 
    mutate(
        last_pres = if_else(n_pres %% 2 == 1, "first", "second"),
        trial_id = row_number(),
        name=wid
    ) %>%
    filter(response_type == "correct")

noisify_strength = function(data, noise_sd) {
    data %>% mutate(
        strength_first = zscore(strength_first + rnorm(n(), sd=noise_sd)),
        strength_second = zscore(strength_second + rnorm(n(), sd=noise_sd)),
        rel_strength = strength_first - strength_second,
        name = paste(name, noise_sd),
        noise_sd = noise_sd,
    )
}

df = bind_rows(
    df %>% noisify_strength(0),
    df %>% noisify_strength(1),
    # df %>% noisify_strength(2),
    # df %>% noisify_strength(4),
) %>% filter(invtemp == 1)

lookup = distinct(df, name, miss_cost, prior, threshold, switch_cost, invtemp, noise_sd)
lookup

# %% --------

df %>% 
    ggplot(aes(rel_strength, int(choose_first), color=factor(noise_sd))) +
    geom_smooth(method = "glm", method.args = list(family = "binomial"), formula=y~x)

fig()

df %>% group_by(miss_cost, prior, threshold, switch_cost, invtemp, noise_sd) %>% 
    summarise(mean(rt), mean(n_pres), sd(n_pres))

raw_df %>% group_by(miss_cost, prior, threshold, switch_cost, invtemp) %>% 
    summarise(mean(correct))

# %% --------

long = df %>% 
    # group_by(name) %>%
    # slice_sample(n=1000) %>% 
    make_fixations

avg_ptime = long %>% group_by(name,wid) %>% 
    filter(last_fix == 0) %>% 
    summarise(duration_mean=mean(duration), duration_sd=sd(duration))

# %% ==================== Plot ====================

fixdata = long %>% 
    ungroup() %>% 
    filter(presentation < 4 & last_fix == 0) %>% 
    left_join(avg_ptime) %>% 
    mutate(duration = (duration - duration_mean) / duration_sd) %>% 
    left_join(select(df, name, trial_id, strength_first, strength_second)) %>% 
    mutate(
        fixated = if_else(fix_first==1, strength_first, strength_second),
        nonfixated = if_else(fix_first==1, strength_second, strength_first),
        relative = if_else(fix_first==1, rel_strength, -rel_strength)
    ) %>% 
    pivot_longer(c(fixated, nonfixated, relative), names_to="cue", values_to="strength", names_prefix="") %>% 
    group_by(name, presentation, cue) %>% 
    select(wid, strength, duration)

models = fixdata %>% 
    group_modify(function(data, grp) {
        # print(grp)
        model = if (grp$name == "Human") {
            lmer(duration ~ strength + (strength|wid), data=data)
        } else {
            lm(duration ~ strength, data=data)
        }
        tibble(model=list(model))
    })

preds = models %>% rowwise() %>% summarise(
    tibble(ggpredict(model, "strength [n=100]"))
)

bindata = fixdata %>% 
    mutate(x = midbins(strength, seq(-2.5, 2.5, 1))) %>% 
    group_by(x, .add=T) %>%
    summarise(mean_cl_normal(duration))


fixdata %>% 
    group_by(name) %>% 
    summarise(mean(strength))

# %% ==================== Plot separate ====================

prep = . %>% 
    left_join(lookup) %>% 
    filter( miss_cost == 100 &
            prior == "(1, 1)" &
            threshold == 60 &
            switch_cost == 2 &
            # invtemp == .3 &
            T
    ) %>% 
    filter(cue != "relative" & between(x, -2.5, 2.5)) %>% 
    filter(!(presentation == 1 & cue == "nonfixated")) %>% 
    mutate(Cue=factor(cue, c("fixated", "nonfixated"), c("Fixated", "Non-Fixated")))

ggplot(prep(preds), aes(x, predicted, group=Cue, color=Cue)) +
    geom_ribbon(aes(ymin=conf.low, ymax=conf.high), alpha=0.1, color=F) +
    geom_line() +
    geom_pointrange(aes(y=y, ymin=ymin, ymax=ymax), prep(bindata), size=.2) +
    facet_grid(presentation~noise_sd) +
    labs(x="Cue Strength", y="Fixation Duration") + scale_colour_manual(
        values=c(
            "#57BBF4",
            "#F5D126"
        ), aesthetics=c("fill", "colour")
     ) + coord_cartesian(ylim=c(-1, 1.2))

fig("", 6, 4)

# %% ==================== Plot relative ====================


# %% --------
    # miss_cost=[0, 100],
    # prior=[(1,1), (2,4), (2,6)],
    # threshold = 30:15:60,
    # switch_cost = 0:1:2,
    # invtemp = [.1, .3, 1]

prep = . %>% 
    left_join(lookup) %>% 
    filter( miss_cost == 100 &
            prior == "(1, 1)" &
            threshold == 60 &
            switch_cost == 2 &
            # invtemp == .3 &
            T
    ) %>% filter(cue == "relative" & between(x, -2.5, 2.5) & presentation > 1)


ggplot(prep(preds), aes(x, predicted)) +
    geom_ribbon(aes(ymin=conf.low, ymax=conf.high), alpha=0.1) +
    geom_line() +
    geom_pointrange(aes(y=y, ymin=ymin, ymax=ymax), prep(bindata), size=.2) +
    facet_grid(presentation~noise_sd) +
    labs(x="Relative Strength of Fixated Cue", y="Fixation Duration") + coord_cartesian(ylim=c(-1, 1.2))

fig("tmp", 7, 7)

# %% --------
