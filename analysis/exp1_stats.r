suppressPackageStartupMessages(source("stats_base.r"))
EXP_NAME = opt_get("exp_name", default="exp1")
write_tex = tex_writer(glue("stats/{EXP_NAME}"))

# %% ==================== load data ====================

pretest = read_csv('../data/processed/exp1/pretest.csv', col_types = cols())

all_trials = load_human(EXP_NAME, "all_trials")

trials = load_human(EXP_NAME, "trials") %>% 
    mutate(
        rt = rt / 1000,
        skip=response_type=="empty", correct=response_type=="correct",
        response_type = recode_factor(response_type, "correct" = "Recalled", "empty" = "Skipped")
    )


participants = load_human('exp1', 'participants')

df = load_model_human("sep11", "exp1", "trials", c("optimal", "flexible"))

# %% ==================== accuracy ====================

props = all_trials %>%
    count(pretest_accuracy, response_type) %>%
    group_by(pretest_accuracy) %>%
    mutate(prop = n / sum(n)) %>%
    select(-n) %>%
    pivot_wider(names_from=response_type, values_from=prop)

props %>%
    mutate(pretest_accuracy = fmt_percent(pretest_accuracy)) %>%
    column_to_rownames(var="pretest_accuracy") %>%
    kbl(format="latex", booktabs=T, digits=3, escape=F) %>%
    gsub('\\\\_', ' ', .) %>%
    gsub('0\\.', '.', .) %>%
    write_tex("response_table", format=F)

props %>%
    rowwise() %>% group_walk(~ with(.x,
        write_tex("{correct:.3}", "acc{pretest_accuracy}", )
    ))

# %% --------

accuracy_table = function(xvar, format_xvar=identity) {
    props = df %>%
        mutate(correct = (response_type == "correct")) %>%
        count(name, {{xvar}}, correct) %>%
        group_by(name) %>%
        mutate(prop = n / sum(n))

    pretest_rates = props %>%
        group_by(name, {{xvar}}) %>%
        summarise(total=sum(prop))

    acc_rates = df %>%
        mutate(correct = (response_type == "correct")) %>%
        group_by(name, {{xvar}}) %>%
        summarise(acc=mean(correct))

    left_join(pretest_rates, acc_rates) %>%
        ungroup() %>%
        mutate(value = fmt("{acc:.3} ({total:.3})"), .keep="unused") %>%
        mutate(xvar = format_xvar({{xvar}}), .keep="unused") %>%
        pivot_wider(names_from=xvar, values_from=value) %>%
        column_to_rownames(var="name") %>%
        kbl(format="latex", booktabs=T, digits=3, escape=F) %>%
        gsub('_', ' ', .) %>%
        gsub('0\\.', '.', .)
}
accuracy_table(pretest_accuracy, fmt_percent) %>% write_tex("accuracy_table", format=F)
accuracy_table(judgement) %>% write_tex("accuracy_table_conf", format=F)

# %% --------

df %>%
    filter(name == "Human") %>%
    count(name, response_type, judgement) %>%
    group_by(response_type) %>%
    mutate(prop = n / sum(n), .keep="unused") %>%
    ungroup() %>%
    pivot_wider(names_from=judgement, values_from=prop) %>%
    select(-name) %>%
    mutate(response_type = recode_factor(response_type, "correct" = "Recalled", "empty" = "Skipped")) %>%
    column_to_rownames(var="response_type") %>%
    kbl(format="latex", booktabs=T, digits=3, escape=F) %>%
    gsub('_', ' ', .) %>%
    gsub('0\\.', '.', .) %>% write_tex("confidence_table", format=F)

# %% --------

df %>%
    count(name, response_type, judgement) %>%
    group_by(name, response_type) %>%
    mutate(prop = n / sum(n), .keep="unused") %>%
    ungroup() %>%
    pivot_wider(names_from=response_type, values_from=prop) %>%
    mutate(value = fmt("{correct:.3} / {empty:.3}"), .keep="unused") %>%
    pivot_wider(names_from=judgement, values_from=value) %>%
    # mutate(response_type = recode_factor(response_type, "correct" = "Recalled", "empty" = "Skipped")) %>%
    column_to_rownames(var="name") %>%
    kbl(format="latex", booktabs=T, digits=3, escape=F) %>%
    gsub('_', ' ', .) %>%
    gsub('0\\.', '.', .) %>%
    write_tex("confidence_table", format=F)


# %% --------

props = all_trials %>%
    count(pretest_accuracy, response_type) %>%
    mutate(prop = n / sum(n), .keep="unused") %>%
    pivot_wider(names_from=response_type, values_from=prop)

props %>%
    select(-pretest_accuracy) %>%
    summarise(across(everything(), sum)) %>%
    pivot_longer(everything(), names_to="name", values_to="value", names_prefix="") %>%
    rowwise() %>% group_walk(~ with(.x,
        fmt_percent(value) %>% write_tex("props/{name}")
    ))

props %>%
    mutate(pretest_accuracy = fmt_percent(pretest_accuracy)) %>%
    kbl(format="latex", booktabs=T, digits=3, escape=F)  %>%
    gsub('_', ' ', .) %>%
    gsub('0\\.', '.', .)

# %% ==================== judgement accuracy ====================

trials %>% 
    filter(skip) %>% 
    mutate(rt = scale(rt), pretest_accuracy = scale(pretest_accuracy)) %>% 
    with(lmer(judgement ~ rt + pretest_accuracy + (rt + pretest_accuracy | wid), data=.)) %>% 
    write_model("judgement")

# %% ==================== reaction times ====================

trials %>% 
    filter(skip) %>% 
    regress(rt ~ judgement) %>% 
    write_model("rt_skip1")

trials %>% 
    filter(skip) %>% 
    regress(rt ~ pretest_accuracy) %>%
    write_model("rt_skip2")

trials %>% 
    filter(correct) %>% 
    regress(rt ~ judgement) %>% 
    write_model("rt_correct1")

trials %>% 
    filter(correct) %>% 
    regress(rt ~ pretest_accuracy) %>%
    write_model("rt_correct2")

# %% ==================== sequential effects ====================

sdata = pretest %>%
    mutate(correct = 1*(response_type == "correct")) %>%
    group_by(wid,word) %>%
    mutate(presentation = row_number()) %>%
    select(wid, word, presentation, correct) %>%
    pivot_wider(names_from=presentation, values_from=correct, names_prefix="pretest") %>%
    right_join(trials)

sdata %>%
    glm(correct ~ pretest1 + pretest2, data=., family=binomial)
    summary

# %% --------

sdata %>%
    ungroup() %>%
    filter(pretest_accuracy == 0.5) %>%
    with(mean(pretest2))

# %% --------

sdata %>%
    ungroup() %>%
    filter(pretest_accuracy == 0.5) %>%
    group_by(pretest2) %>%
    summarise(mean(correct))


# %% --------

trials %>% count(response_type)