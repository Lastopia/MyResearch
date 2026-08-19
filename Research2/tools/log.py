import time


LOG_WIDTHS = [8, 6, 30, 24, 30, 18, 18, 16, 16]


def stage_title(name):
    print(f"========== [{name.upper()} START] ==========")


def now():
    return time.time()


def elapsed_seconds(start):
    return int(time.time() - start)


def _cell(value, width):
    text = "" if value is None else str(value)
    if len(text) > width:
        text = text[: max(0, width - 1)] + "~"
    return text.ljust(width)


def train_line(stage, gpu, alias, step, train_loss=None, valid_loss=None, seconds=None, total_seconds=None, mem=None, vram=None):
    train_text = None if train_loss is None else f"train_loss={train_loss:.2f}"
    valid_text = None if valid_loss is None else f"valid_loss={valid_loss:.2f}"
    if seconds is not None and total_seconds is not None:
        time_text = f"time={seconds}s/{total_seconds}s"
    elif seconds is not None:
        time_text = f"time={seconds}s"
    else:
        time_text = None
    values = [
        f"[{stage}]",
        gpu,
        alias,
        f"step={step}",
        train_text,
        valid_text,
        time_text,
        None if mem is None else f"mem={mem}",
        None if vram is None else f"vram={vram}",
    ]
    parts = [
        _cell(value, width)
        for value, width in zip(values, LOG_WIDTHS)
    ]
    print(" | ".join(parts), flush=True)


def event_line(stage, gpu=None, alias=None, seed=None, event=None, detail=None):
    alias_text = alias
    if alias_text is not None and seed is not None:
        alias_text = f"{alias_text} s{seed}"
    values = [
        f"[{stage}]",
        gpu,
        alias_text,
        event,
        detail,
        None,
        None,
        None,
        None,
    ]
    parts = [_cell(value, width) for value, width in zip(values, LOG_WIDTHS)]
    print(" | ".join(parts), flush=True)
