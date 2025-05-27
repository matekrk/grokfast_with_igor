import inspect
import logging
import os
from datetime import datetime
from enum import Enum


class LogLevel(Enum):
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"


class SmartLogger:
    """
    A logging class that automatically includes caller information.
    Supports different output modes based on environment.
    """

    def __init__(self,
                 output_mode="auto",  # "console", "file", "server", "auto"
                 log_file="debug.log",
                 server_url=None,
                 include_timestamp=True,
                 shorten_names=True):

        self.output_mode = self._determine_output_mode(output_mode)
        self.log_file = log_file
        self.server_url = server_url
        self.include_timestamp = include_timestamp
        self.shorten_names = shorten_names

        # Setup based on mode
        if self.output_mode == "file":
            self._setup_file_logging()

    def _determine_output_mode(self, mode):
        """Determine output mode based on environment if 'auto' is selected."""
        if mode != "auto":
            return mode

        # Auto-detect based on environment
        if os.getenv("JUPYTER_RUNTIME_DIR") or os.getenv("COLAB_GPU"):
            return "console"  # Jupyter/Colab
        elif os.getenv("PRODUCTION") == "true":
            return "server"  # Production
        else:
            return "console"  # Development

    def _setup_file_logging(self):
        """Setup file logging if needed."""
        logging.basicConfig(
            filename=self.log_file,
            level=logging.DEBUG,
            format='%(message)s'
        )

    def _get_caller_info(self):
        """Get information about the calling function/method."""
        # Go back 2 frames: current -> log method -> actual caller
        frame = inspect.currentframe().f_back.f_back

        function_name = frame.f_code.co_name
        args_info = inspect.getargvalues(frame)
        args = args_info.args

        class_name = None

        if args and args[0] in ('self', 'cls'):
            instance_or_class = args_info.locals[args[0]]

            if args[0] == 'self':
                class_name = instance_or_class.__class__.__name__
            elif args[0] == 'cls':
                class_name = instance_or_class.__name__

        # Shorten names if requested
        if self.shorten_names:
            def _shorten(name):
                if not name:
                    return name
                if name.startswith('torch_'):
                    name = name[6:]
                elif name.startswith('nn_'):
                    name = name[3:]
                name = name.replace('__', '_').rstrip('_')
                return name

            class_name = _shorten(class_name)
            function_name = _shorten(function_name)

        if class_name:
            return f"{class_name}.{function_name}()"
        else:
            return f"{function_name}()"

    def _format_message(self, level, message):
        """Format the log message with caller info and timestamp."""
        caller_info = self._get_caller_info()

        parts = []

        if self.include_timestamp:
            timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
            parts.append(f"[{timestamp}]")

        parts.append(f"[{level.value}]")
        parts.append(f"[{caller_info}]")
        parts.append(message)

        return " ".join(parts)

    def _output_message(self, formatted_message):
        """Output the message based on the configured mode."""
        if self.output_mode == "console":
            print(formatted_message)
        elif self.output_mode == "file":
            logging.info(formatted_message)
        elif self.output_mode == "server":
            self._send_to_server(formatted_message)

    def _send_to_server(self, message):
        """Send log message to server (implement based on your needs)."""
        # Placeholder for server logging
        # You could use requests, websockets, etc.
        print(f"[SERVER] {message}")  # Fallback to console for now

    # Main logging methods
    def debug(self, message):
        formatted = self._format_message(LogLevel.DEBUG, message)
        self._output_message(formatted)

    def info(self, message):
        formatted = self._format_message(LogLevel.INFO, message)
        self._output_message(formatted)

    def warning(self, message):
        formatted = self._format_message(LogLevel.WARNING, message)
        self._output_message(formatted)

    def error(self, message):
        formatted = self._format_message(LogLevel.ERROR, message)
        self._output_message(formatted)

    # Convenience method
    def log(self, message, level=LogLevel.INFO):
        """General log method."""
        formatted = self._format_message(level, message)
        self._output_message(formatted)


# Create a global logger instance
logger = SmartLogger()