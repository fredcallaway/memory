suppressPackageStartupMessages(source("stats_base.r"))
EXP_NAME = opt_get("exp_name", default="exp1")
write_tex = tex_writer(glue("stats/{EXP_NAME}"))

# %% ==================== load data ====================

pretest = read_csv(glue('../data/processed/{EXP_NAME}/pretest.csv'), col_types = cols())

all_trials = load_human(EXP_NAME, "all_trials")

trials = load_human(EXP_NAME, "trials") %>% 
    mutate(
        rt = rt / 1000,
        skip=response_type=="empty", correct=response_type=="correct",
        response_type = recode_factor(response_type, "correct" = "Recalled", "empty" = "Skipped")
    )

participants = load_human('exp1', 'participants')

# HACK: hardcoding model run here
df = load_model_human("apr6", EXP_NAME, "trials", c("optimal", "flexible"))

# %% ==================== excluding no-response pretest ====================

pretest %>%
    filter(response_type %in% c("correct", "intrusion")) %>%
    group_by(wid, word) %>%
    summarise(pretest_accuracy = mean(response_type == "correct")) %>%
    right_join(select(trials, -pretest_accuracy)) %>%
    filter(skip) %>%
    regress(rt ~ pretest_accuracy) %>%
    write_model("rt_skip_intrusion")


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

# %% ==================== all response type frequencies ====================

all_trials %>%
    count(response_type) %>%
    mutate(prop = n / sum(n)) %>%
    rowwise() %>%
    group_walk(~ with(.x,
        write_tex("{100 * prop:.1}\\%", "response_prop/{response_type}")
    ))

all_trials %>%
    count(pretest_accuracy, response_type) %>%
    group_by(pretest_accuracy) %>%
    mutate(prop = n / sum(n)) %>%
    select(-n) %>%
    pivot_wider(names_from=response_type, values_from=prop) %>%
    mutate(pretest_accuracy = fmt_percent(pretest_accuracy)) %>%
    column_to_rownames(var="pretest_accuracy") %>%
    kbl(format="latex", booktabs=T, digits=3, escape=F) %>%
    gsub('\\\\_', ' ', .) %>%
    gsub('0\\.', '.', .) %>%
    write_tex("response_table", format=F)

# %% ==================== recall rates (vs. skip) ====================

pretest_rates = df %>%
    mutate(correct = (response_type == "correct")) %>%
    count(name, pretest_accuracy, correct) %>%
    group_by(name) %>%
    mutate(prop = n / sum(n)) %>%
    group_by(name, pretest_accuracy) %>%
    summarise(prop=sum(prop))

pretest_rates %>%
    filter(name == "Human") %>%
    rowwise() %>% group_walk(~ with(.x,
        write_tex("{100*prop:.1}\\%", "pretest_prop/{pretest_accuracy}", )
    ))

acc_rates = df %>%
    mutate(correct = (response_type == "correct")) %>%
    group_by(name, pretest_accuracy) %>%
    summarise(acc=mean(correct))

acc_rates %>%
    filter(name == "Human") %>%
    rowwise() %>% group_walk(~ with(.x,
        write_tex("{100*acc:.1}\\%", "recall/{pretest_accuracy}", )
    ))

left_join(pretest_rates, acc_rates) %>%
    ungroup() %>%
    mutate(value = fmt("{acc:.3} ({prop:.3})"), .keep="unused") %>%
    mutate(xvar = fmt_percent(pretest_accuracy), .keep="unused") %>%
    pivot_wider(names_from=xvar, values_from=value) %>%
    column_to_rownames(var="name") %>%
    kbl(format="latex", booktabs=T, digits=3, escape=F) %>%
    gsub('_', ' ', .) %>%
    gsub('0\\.', '.', .) %>%
    write_tex("recall_table", format=F)

# %% ==================== judgment frequencies ====================

df %>%
    filter(name == "Human") %>%
    count(name, response_type, judgement) %>%
    group_by(response_type) %>%
    mutate(prop = n / sum(n), .keep="unused") %>%
    ungroup() %>%
    pivot_wider(names_from=judgement, values_from=prop) %>%
    select(-name) %>%
    mutate(response_type = recode_factor(response_type,
        "correct" = "Confidence (Recalled)",
        "empty" = "Feeling of Knowing (Skipped)"
    )) %>%
    column_to_rownames(var="response_type") %>%
    kbl(format="latex", booktabs=T, digits=3, escape=F) %>%
    gsub('_', ' ', .) %>%
    gsub('0\\.', '.', .) %>%
    write_tex("judgment_table", format=F)

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
    summarise(acc=mean(correct)) %>%
    rowwise() %>% group_walk(~ with(.x,
        write_tex("{100 * acc:.1}\\%", "sequential/{pretest2}")
    ))

# %% ==================== judgement accuracy ====================

trials %>%
    filter(skip) %>%
    regress(judgement ~ pretest_accuracy) %>%
    write_model("fok")

trials %>% 
    filter(correct) %>%
    regress(judgement ~ pretest_accuracy) %>%
    write_model("confidence")


# %% --------

response_trials = all_trials %>%
    filter(response_type %in% c("correct", "intrusion", "other"))

response_trials %>%
    regress_logistic(correct ~ judgement) %>%
    write_model("confidence_accuracy", logistic=T)

response_trials %>%
    group_by(judgement) %>%
    summarise(acc=mean(correct)) %>%
    rowwise() %>% group_walk(~ with(.x,
        write_tex("{100 * acc:.1}\\%", "correct_prop/{judgement}")
    ))