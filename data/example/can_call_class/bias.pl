head_pred(canCallClass,2).
body_pred(class,1).
body_pred(method,1).
body_pred(containsMethod,2).
body_pred(callsMethod,2).

type(canCallClass,(class,class)).
type(class,(class,)).
type(method,(method,)).
type(containsMethod,(class,method)).
type(callsMethod,(method,method)).

max_vars(4).
max_body(3).
max_clauses(1).
