suppressPackageStartupMessages(source("stats_base.r"))
EXP_NAME = opt_get("exp_name", default="exp2")
write_tex = tex_writer(glue("stats/{EXP_NAME}"))

df = load_model_human("sep11", "exp2", "trials", c("fixed_optimal", "flexible"))


# %% ==================== load data ====================

pretest = read_csv('../data/processed/exp2/pretest.csv', col_types = cols())

all_trials =  load_human("exp2", "trials_witherr")

trials = load_human("exp2", "trials") %>% 
    mutate(rel_pretest_accuracy = pretest_accuracy_first - pretest_accuracy_second)

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
        ),
        fixated = case_when(
            mod(presentation, 2) == 1 ~ pretest_accuracy_first,
            mod(presentation, 2) == 0 ~ pretest_accuracy_second,
        ),
        nonfixated = case_when(
            mod(presentation, 2) == 1 ~ pretest_accuracy_second,
            mod(presentation, 2) == 0 ~ pretest_accuracy_first,
        ),
        relative = fixated - nonfixated
    )

# %% --------
X = df %>%
    mutate(trial_type = glue("{pretest_accuracy_first} {pretest_accuracy_second}"))

# %% --------

X %>%
    filter(trial_type %in% c("1 1", "0 0", "0.5 0.5")) %>%
    group_by(name, trial_type) %>%
    summarise(mean(choose_first & n_pres == 1))
    # summarise(mean(rt), mean(n_pres), mean(choose_first, na.rm=T))

X %>%
    filter(trial_type == "0 0", n_pres < 4) %>%
    count(name, response_type, trial_type, n_pres) %>%
    group_by(name) %>%
    mutate(prop = n/sum(n)) %>%
    select(-n, -trial_type) %>%
    pivot_wider(names_from=name, values_from=prop)



startsWith("foo", "f")

# %% ==================== accuracy ====================

all_trials %>% 
    with(mean(response_type == "correct")) %>% 
    fmt_percent %>% 
    write_tex("accuracy")


all_trials %>%
    count(choose_first, )

# %% --------

props = df %>%
    mutate(correct = (response_type == "correct")) %>%
    count(name, pretest_accuracy, correct) %>%
    group_by(name) %>%
    mutate(prop = n / sum(n))

pretest_rates = props %>%
    group_by(name, pretest_accuracy) %>%
    summarise(total=sum(prop))

acc_rates = df %>%
    mutate(correct = (response_type == "correct")) %>%
    group_by(name, pretest_accuracy) %>%
    summarise(acc=mean(correct))

left_join(pretest_rates, acc_rates) %>%
    ungroup() %>%
    mutate(value = fmt("{acc:.3} ({total:.3})"), .keep="unused") %>%
    mutate(pretest_accuracy = fmt_percent(pretest_accuracy)) %>%
    pivot_wider(names_from=pretest_accuracy, values_from=value) %>%
    column_to_rownames(var="name") %>%
    kbl(format="latex", booktabs=T, digits=3, escape=F) %>%
    gsub('_', ' ', .) %>%
    gsub('0\\.', '.', .) %>%
    write_tex("accuracy_table", format=F)

# %% ==================== overall proportion ====================

trials %>%
    filter(n_pres > 1) %>% 
    mutate(prop_first = total_first / (total_first + total_second)) %>% 
    regress(prop_first ~ rel_pretest_accuracy) %>% 
    write_model("prop_first")

# %% ==================== last fixation confound ====================

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
    # filter(n_pres > 1) %>% 
    with(mean(choose_first == mod(n_pres, 2))) %>% 
    fmt_percent %>% 
    write_tex("prop_choose_last")

# %% --------

fixations %>% 
    filter(n_pres > 1) %>% 
    group_by(name, wid, trial_id, is_final=presentation==n_pres) %>% 
    summarise(x=sum(duration)) %>%
    mutate(prop = x / sum(x)) %>%
    filter(is_final) %>% 
    collapse_participants(mean, prop) %>%
    with(mean(prop)) %>% 
    fmt_percent %>% 
    write_tex("prop_last_duration")

# %% ==================== duration by number and finality ====================

fixations %>% 
    mutate(final = 1 * (presentation == n_pres)) %>%
    regress(duration ~ final + presentation) %>% 
    write_model("duration")

fixations %>% 
    mutate(final = 1*(presentation == n_pres)) %>% 
    regress(duration ~ final) %>% 
    write_model("duration_final")

# %% ==================== nonfinal durations ====================

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

nonfinal %>% 
    regress(duration ~ fixated) %>% 
    write_model("nonfinal")

nonfinal %>% 
    filter(presentation > 1) %>% 
    regress(duration ~ nonfixated) %>% 
    write_model("nonfinal")

# %% ==================== rational commitment ====================

fixations %>% 
    mutate(bad = relative < 0) %>% 
    regress_logistic(bad ~ duration, data=.) %>% 
    write_model("prop_bad", logistic=T)



