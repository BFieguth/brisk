from sklearn import metrics

def create_metric(func, name, abbr=None):
    scorer = metrics.make_scorer(func)
    return {
        "func": func,
        "scorer": scorer,
        "abbr": abbr,
        "display_name": name
    }
