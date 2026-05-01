BAR = "=============="


def print_with_bar(log_msg):
    log_msg = BAR + log_msg + BAR
    if "End" in log_msg:
        log_msg += "\n"
    print(log_msg)


def get_table_parameters(model, table):
    for name, param in model.named_parameters():
        if table in name and "gnn" not in name:
            yield param
        elif table == "gnn" and "gnn" in name:
            yield param
