import logging
import os
import sys


class TeeStream:
    """
    Forwards every write to multiple streams simultaneously.

    Used to redirect sys.stdout and sys.stderr so that print() calls,
    tqdm progress bars, and any other direct writes to those streams
    are captured in the log file as well as appearing on the terminal.
    """

    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for s in self.streams:
            s.write(data)

    def flush(self):
        for s in self.streams:
            s.flush()

    # Proxy attributes that some libraries (e.g. tqdm, ipython) may query
    def fileno(self):
        return self.streams[0].fileno()

    def isatty(self):
        return self.streams[0].isatty()

    @property
    def encoding(self):
        return getattr(self.streams[0], 'encoding', 'utf-8')


def setup_logger(log_file_path):
    """
    Set up the logger and redirect sys.stdout / sys.stderr so that
    *all* console output — logger calls, print() statements, and tqdm
    progress bars — is also written to the log file.

    The original streams are stored on the returned logger object so
    that callers can restore them when the run finishes:

        logger = setup_logger(path)
        try:
            ...
        finally:
            logger.restore_streams()

    Args:
        log_file_path (str): Path to the log file.

    Returns:
        logger: Configured logger object with a ``restore_streams()`` method.
    """
    os.makedirs(os.path.dirname(log_file_path), exist_ok=True)

    # Open the log file with line buffering so output appears promptly
    log_fh = open(log_file_path, 'w', buffering=1, encoding='utf-8')

    # Save originals before replacing
    original_stdout = sys.stdout
    original_stderr = sys.stderr

    # Tee both streams into the log file
    sys.stdout = TeeStream(original_stdout, log_fh)
    sys.stderr = TeeStream(original_stderr, log_fh)

    # Build logger – a single console handler is enough because sys.stdout
    # is now a TeeStream that already fans out to the terminal and the file.
    logger = logging.getLogger('logger')
    logger.setLevel(logging.INFO)
    logger.handlers = []

    formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
    )
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # Attach cleanup state to the logger for later restoration
    logger._log_fh          = log_fh
    logger._original_stdout = original_stdout
    logger._original_stderr = original_stderr

    def restore_streams():
        """Flush and close the log file, then restore sys.stdout/stderr."""
        sys.stdout = logger._original_stdout
        sys.stderr = logger._original_stderr
        try:
            logger._log_fh.flush()
            logger._log_fh.close()
        except Exception:
            pass

    logger.restore_streams = restore_streams

    logger.info("=" * 80)
    logger.info("Starting Process")
    logger.info(f"Log file: {log_file_path}")
    logger.info("=" * 80)

    return logger
