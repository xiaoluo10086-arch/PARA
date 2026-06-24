# Minimal `canCallClass` Example

The accepted rule is:

```prolog
canCallClass(A,B) :-
    containsMethod(A,M1),
    callsMethod(M1,M2),
    containsMethod(B,M2).
```

For `canCallClass(order_service,payment_client)`, PARA grounds the rule with:

```text
containsMethod(order_service,submit_order)
callsMethod(submit_order,charge_payment)
containsMethod(payment_client,charge_payment)
```

This fixture tests proof construction only. It does not call an LLM or claim to
reproduce the full paper experiment.
