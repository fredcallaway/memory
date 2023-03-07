source("setup.r")
library(formula.tools)
library(kableExtra)

tidy = function(model, ...) {
    d = broom.mixed::tidy(model, conf.int=T, ...)
    if (typeof(model) == "list") {
        d$df = model$df
    }
    d
}

regress = function(data, form, standardize=F) {
    preds = deparse(rhs(form))
    data = tibble(data)
    if (standardize) {
        for (k in get.vars(form)) {
            data[[k]] = zscore(data[[k]])
        }
    }
    form = as.formula(glue("{form} + ({preds} | wid)"))
    lmer(form, data=data)
}

regress_logistic = function(data, form, standardize=F) {
    preds = deparse(rhs(form))
    data = tibble(data)
    if (standardize) {
        for (k in get.vars(form)) {
            data[[k]] = zscore(data[[k]])
        }
    }
    form = as.formula(glue("{form} + ({preds} | wid)"))
    glmer(form, data=data, family=binomial)
}

regression_tex = function(logistic=F, standardized=T) {
    beta = if(standardized) "$\\beta = {estimate:.3}$" else "$B = {estimate:.3}$"
    ci = "95\\% CI [{conf.low:.3}, {conf.high:.3}]"
    stat = if(logistic) "$z={statistic:.2}$" else "$t({df:.1})={statistic:.2}$"
    p = "${pval(p.value)}$"
    paste(beta, ci, stat, p, sep=", ")
}

write_model = function(model, name, logistic=F, standardized=F) {
    model %>% 
        tidy %>% 
        filter((term != "(Intercept)") & (are_na(effect) | effect == "fixed")) %>% 
        rowwise() %>% group_walk(~ with(.x,
            regression_tex(logistic, standardized) %>% 
            write_tex("{name}/{term}")
        ))        
}

fmt_percent = function(prop) glue("{round(100 * prop)}\\%")
