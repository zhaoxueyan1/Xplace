import logging
import os
import sys
import json
import datetime


class CustomFormatter(logging.Formatter):
    white = "\x1b[37;21m"
    blue = "\x1b[34;21m"
    yellow = "\x1b[33;21m"
    green = "\x1b[32;21m"
    red = "\x1b[31;21m"
    bold_red = "\x1b[31;1m"
    reset = "\x1b[0m"

    # time_format = "[%(asctime)s.%(msecs)03d] "
    # FIXME: An elapsed time gap exists between the C++ Timer and Python Timer,
    #        but I don't know how to resolve it...
    time_format = "[%(relativeCreatedSecond)4d.%(relativeCreatedMSecond)03d] "
    debug_msg = " (%(module)s.py L%(lineno)d) %(msg)s"
    FORMATS = {
        logging.DEBUG: time_format + blue + "DEBUG" + reset + debug_msg,
        logging.INFO: time_format + "%(msg)s",
        logging.WARNING: time_format + yellow + "WARNING" + reset + " %(msg)s",
        logging.ERROR: time_format + red + "ERROR" + reset + " %(msg)s",
        logging.CRITICAL: time_format + bold_red + "CRITICAL" + reset + " %(msg)s",
    }

    def __init__(self):
        super().__init__(datefmt="%H:%M:%S")

    def format(self, record):
        record.relativeCreatedSecond = record.relativeCreated / 1000
        record.relativeCreatedMSecond = record.relativeCreated % 1000
        self._style._fmt = self.FORMATS.get(record.levelno)
        return logging.Formatter.format(self, record)


def setup_logger(args, sys_argv) -> logging.Logger:
    res_root = os.path.join(args.result_dir, args.exp_id)
    log_file_path = os.path.join(res_root, args.log_dir, args.log_name)
    if not os.path.exists(os.path.dirname(log_file_path)):
        os.makedirs(os.path.dirname(log_file_path))

    formatter = CustomFormatter()
    file_handler = logging.FileHandler(log_file_path, mode="w")
    file_handler.setFormatter(formatter)
    screen_handler = logging.StreamHandler(stream=sys.stdout)
    screen_handler.setFormatter(formatter)
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    if args.cpp_log_level == 0 or args.verbose_cpp_log:
        logger.setLevel(logging.DEBUG)
    logger.addHandler(file_handler)
    logger.addHandler(screen_handler)

    logger.info("Command line: python " + " ".join(sys_argv))
    logger.info("log file at {}".format(log_file_path))
    logger.info("")
    for arg in vars(args):
        logger.info("{}: {}".format(arg, getattr(args, arg)))
    logger.info("")

    logging.getLogger("matplotlib.font_manager").disabled = True
    return logger


def _runtime_params_path(args, filename="effective_runtime_params.json"):
    res_root = os.path.join(args.result_dir, args.exp_id)
    output_path = os.path.join(res_root, args.log_dir, filename)
    output_dir = os.path.dirname(output_path)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    return output_path


def _to_jsonable(value):
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(key): _to_jsonable(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(item) for item in value]
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass
    if hasattr(value, "tolist"):
        try:
            return value.tolist()
        except Exception:
            pass
    return str(value)


def _load_runtime_params(path):
    if not os.path.exists(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as fp:
            payload = json.load(fp)
        return payload if isinstance(payload, dict) else {}
    except Exception:
        return {}


def _write_runtime_params(path, payload):
    with open(path, "w", encoding="utf-8") as fp:
        json.dump(payload, fp, indent=2, sort_keys=True)
        fp.write("\n")


def save_effective_runtime_params(args, extra_sections=None, filename="effective_runtime_params.json"):
    path = _runtime_params_path(args, filename=filename)
    payload = _load_runtime_params(path)
    payload["effective_args"] = _to_jsonable({arg: getattr(args, arg) for arg in vars(args)})
    if extra_sections:
        for key, value in extra_sections.items():
            payload[key] = _to_jsonable(value)
    payload["updated_at"] = datetime.datetime.now().isoformat()
    _write_runtime_params(path, payload)
    return path


def append_gr_run_params(args,
                         run_tag,
                         gr_params,
                         extra=None,
                         filename="effective_runtime_params.json"):
    path = _runtime_params_path(args, filename=filename)
    payload = _load_runtime_params(path)
    payload["effective_args"] = _to_jsonable({arg: getattr(args, arg) for arg in vars(args)})
    gr_runs = payload.get("gr_runs")
    if not isinstance(gr_runs, list):
        gr_runs = []

    entry = {
        "run_tag": run_tag,
        "gr_params": _to_jsonable(gr_params),
        "recorded_at": datetime.datetime.now().isoformat(),
    }
    if extra:
        for key, value in extra.items():
            entry[key] = _to_jsonable(value)

    gr_runs.append(entry)
    payload["gr_runs"] = gr_runs
    payload["updated_at"] = datetime.datetime.now().isoformat()
    _write_runtime_params(path, payload)
    return path
