pos(canCallClass(order_service,payment_client)).
pos(canCallClass(order_service,user_service)).

neg(canCallClass(payment_client,user_service)).
neg(canCallClass(logger,payment_client)).
