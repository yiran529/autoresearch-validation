"""Save inherited PowerSGD slots as well as dynamically attached state fields."""


def snapshot_hook(state, cpu):
    values = dict(vars(state)) if hasattr(state, '__dict__') else {}
    for cls in reversed(type(state).__mro__):
        slots = cls.__dict__.get('__slots__', ())
        if isinstance(slots, str):
            slots = (slots,)
        for name in slots:
            if name not in {'__dict__', '__weakref__'} and hasattr(state, name):
                values[name] = getattr(state, name)
    # These process-local objects are recreated by hook attachment.
    excluded = {'process_group', 'param_state', 'param_to_name', 'excluded_parameter_ids'}
    return {name: cpu(value) for name, value in values.items() if name not in excluded}
