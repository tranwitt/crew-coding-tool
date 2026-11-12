"""Plain-text run logs: tee console output into a timestamped .log file."""
import re
import sys
import traceback

ANSI = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")
RULE = '=' * 72


class ConsoleCapture:
    """Tee CrewAI output to the terminal and an in-memory plain-text transcript."""
    def __init__(self, stream, chunks):
        self.stream, self.chunks = stream, chunks

    def write(self, value):
        self.stream.write(value)
        self.chunks.append(value)
        return len(value)

    def flush(self):
        self.stream.flush()

    def isatty(self):
        return self.stream.isatty()

    def __getattr__(self, name):
        return getattr(self.stream, name)


def clean_console(value):
    """Remove terminal styling and normalize carriage-return progress updates."""
    value = ANSI.sub('', value)
    return value.replace('\r\n', '\n').replace('\r', '\n')


def capture_run(log_path, metadata, callback):
    """Run ``callback`` while capturing its console transcript into ``log_path``.

    The log is written even when the callback raises; the exception propagates.
    """
    log_path.parent.mkdir(parents=True, exist_ok=True)
    old_out, old_err = sys.stdout, sys.stderr
    chunks = []
    error = None
    sys.stdout = ConsoleCapture(old_out, chunks)
    sys.stderr = ConsoleCapture(old_err, chunks)
    try:
        return callback()
    except Exception:
        error = traceback.format_exc()
        traceback.print_exc()
        raise
    finally:
        sys.stdout, sys.stderr = old_out, old_err
        with log_path.open('w', encoding='utf-8') as log:
            log.write('CREWAI CODING RUN\n' + RULE + '\n')
            for key, value in metadata.items():
                log.write(f'{key}: {str(value).replace(chr(10), " ")}\n')
            log.write('\n' + RULE + '\nCREW EXECUTION TRANSCRIPT\n' + RULE + '\n')
            log.write(clean_console(''.join(chunks)).rstrip() + '\n')
            if error:
                log.write('\nEXECUTION ERROR\n' + '-' * 72 + '\n' + clean_console(error).rstrip() + '\n')


def append_section(log_path, title, body):
    """Append a titled section to an existing run log; a missing log is ignored."""
    if not log_path.is_file():
        return
    with log_path.open('a', encoding='utf-8') as log:
        log.write(f'\n{title}\n' + '-' * 72 + '\n' + str(body).rstrip() + '\n')
