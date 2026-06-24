class(order_service).
class(payment_client).
class(user_service).
class(logger).

method(submit_order).
method(charge_payment).
method(load_user).
method(write_log).

containsMethod(order_service,submit_order).
containsMethod(payment_client,charge_payment).
containsMethod(user_service,load_user).
containsMethod(logger,write_log).

callsMethod(submit_order,charge_payment).
callsMethod(submit_order,load_user).
