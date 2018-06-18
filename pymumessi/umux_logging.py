"""
Author : Joshua Montgomery <Joshua.Montgomery@mail.mcgill.ca>
Date   : July 17, 2014

This module contains Logging-related classes which define main and children
python logging objects, and specify the custom formatting used to write logging
messages to the console, to disk, and to a GrayLogger mongodb via UDP GELF
packets.

In particular:
 - The LoggingManager class is used to serve main and child logging instances
   throughout the code-base. It insantiates logging objects with the correct
   handlers
 - The DfmuxFormatter class is a subclass of the python Logging Formatter
   class.  It formats and includes within the logging message additional
   persistant fields generated from within algorithms elsewhere and passed to
   the logging objects, as well as colorizing console output.

The mainlogger object has two handlers by default.
In general it should only be called directly if the message is not associated
with a specific target object, or from within an
algorithm that is being parallelized over.
    - Console (prints logging messages to stdout, colorized)
    - FileLogger (appends logging messages to a single .txt file every day --
      these are automatically generated every day)

The childlogger object is most often what is instantiated from within
algorithms to log activity by algorithm and target object. It passes all of its
messages up to the mainlogger handlers, but additionally has one handler of its
own which produces discrete .txt logs for each algorithm/target_object
instance. These logs are far easier to search and parse than the master day
running log:
    - The FileLogger (child) handler makes a new, separate .txt log file that
      lives in a hierarchical directory Date->Algorithm->Time_Object

If you don't have a PYDFMUX_LOG_DIR environment variable you will get an
exception asking for you to make one.

TODO:
    1. Consistency checks in the presence of parallelization
    2. **kwargs support for the extra_fields dictionaries
    3. Make a method that allows for quickly and easily changing the
       console logger level

Examples:
    >>> import pydfmux.core.utils.logging_utils as LU
    >>> LM = LU.LoggingManager()
    >>> mainlogger = LM.get_main_logger()
    >>> mainlogger.info('This is a main-logger message', extra={'extra_field_key':'extra_field_value'})
    >>>     '2014-07-28 17:23:17 | INFO | INPUT_FILE.py | INPUT_MODULE | This is a main-logger message | extra_field_key: extra_field_value'
    >>> childlogger = LM.get_child_logger(target=TARGET_OBJECT)
    >>> childlogger.info('This is a logging message', extra={'extra_field_key':'extra_field_value'})
    >>>     '2014-07-28 17:24:17 | INFO | MACRO | INPUT_FILE.py | INPUT_MODULE | 'TARGET_OBJECT' | This is a logging message | extra_field_key: extra_field_value'"""

import logging
import sys
import os
import copy
import time
import datetime
import warnings
import pymumessi

# add a NOTICE logging level

logging.NOTICE = 25
logging.addLevelName(logging.NOTICE, 'NOTICE')

def logger_notice(self, msg, *args, **kwargs):
    """
    Log 'msg % args' with severity 'NOTICE'.
    """
    if self.isEnabledFor(logging.NOTICE):
        self._log(logging.NOTICE, msg, args, **kwargs)

logging.Logger.notice = logger_notice

def root_notice(msg, *args, **kwargs):
    """
    Log a message with severity 'NOTICE' on the root logger.
    """
    if len(logging.root.handlers) == 0:
        logging.basicConfig()
    logging.root.notice(msg, *args, **kwargs)

logging.notice = root_notice

# timezone handling, independent of logging yaml
# because that config doesn't understand extra options

def _get_timezone(timezone):
    """Convert input to datetime.tzinfo object for formatting
    timestamps.  Argument can be a string timezone name or a
    specific tzinfo instance.  If None or 'local', returns the
    local timezone if found."""

    # explicit timezone specified as tzinfo object
    if isinstance(timezone, datetime.tzinfo):
        return timezone

    try:
        import pytz
    except ImportError:
        has_pytz = False
    else:
        has_pytz = True

    if str(timezone).lower() == 'local':
        timezone = None

    # explicit timezone specified as string name
    if timezone:
        # can't handle timezones
        if not has_pytz:
            return None
        return pytz.timezone(timezone)

    # no timezone specified, use local if possible
    try:
        import tzlocal
    except ImportError:
        if not has_pytz:
            return None
        try:
            with open('/etc/localtime', 'rb') as f:
                return pytz.tzfile.build_tzinfo('local', f)
        except IOError:
            return None
    else:
        return tzlocal.get_localzone()

# default to UTC time
_logging_timezone = _get_timezone('UTC')

def set_logging_timezone(timezone):
    """
    Set the global default timezone for data output directories and
    log entries.

    Options are:

    'UTC' : universal time (default)
    'local' or None : attempt to figure out the system's local timezone
        NB: this is best used with the `tzlocal` python package.
    string name : a known timezone string name, e.g. 'US/Eastern'
    datetime.tzinfo instance: an existing timezone object
    """
    global _logging_timezone
    _logging_timezone = _get_timezone(timezone)


class LoggingManager(object):
    """A custom class that makes the MainLogger and childlogging objects,
    adds default handlers for the console and file_logging, and defines the basic
    logging message format
    See the Module docstring for more detailed information about usage.

    Args:
        logdir: The subdirectory name where logs should live (subdirectory to
            PYDFMUX_LOG_DIR). This is useful to set to keep development logs
            apart from production logs. By default it is empty.

        log_colors: a dictionary of default logging colors, to be passed
            to child DfmuxFormatter instances.  If None, the default log colors
            are used.

        timezone: the timezone to be used for data output directory timestamps
            and log file entries.  Can be 'UTC', 'local', a string timezone name
            or `datetime.tzinfo` instance. If None, the global pydfmux default
            is used.

    Examples:
        >>> import pydfmux.core.utils.logging_utils as LU
        >>> LM = LU.LoggingManager()
        >>> mainlogger = LU.logger
        >>> mainlogger.info('This is a main-logger message', extra={'extra_field_key':'extra_field_value'})
        >>>     '2014-07-28 17:23:17 | INFO | INPUT_FILE.py | INPUT_MODULE | This is a main-logger message | extra_field_key: extra_field_value'
        >>> childlogger = LM.get_child_logger(target=TARGET_OBJECT)
        >>> childlogger.info('This is a logging message', extra={'extra_field_key':'extra_field_value'})
        >>>     '2014-07-28 17:24:17 | INFO | MACRO | INPUT_FILE.py | INPUT_MODULE | 'TARGET_OBJECT' | This is a logging message | extra_field_key: extra_field_value'
        """

    def __init__(self, logdir='', log_colors=None, timezone=None):

        self.BASIC_FORMAT = '%(asctime)s | %(levelname)s | %(filename)s | %(funcName)s | %(message)s'
        self.logger = logging.getLogger()

        # Configure the root logger
        root_formatter = DfmuxFormatter(self.BASIC_FORMAT,
                                        log_colors=log_colors,
                                        timezone=timezone)
        console_handler = logging.StreamHandler(sys.stderr)
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(root_formatter)
        self.logger.addHandler(console_handler)

        file_console_handler = logging.handlers.TimedRotatingFileHandler('pymumessi.log',
                                                                         when='midnight',
                                                                         interval=1,
                                                                         backupCount=90)
        file_console_handler.setLevel(logging.INFO)
        file_console_handler.setFormatter(root_formatter)
        self.logger.addHandler(file_console_handler)
        
        # Figure out where we are putting things (for the child-loggers)
        if logdir:
            self.logdir = logdir
        elif 'PYMUMESSI_OUTDIR' in os.environ:
            self.logdir = os.path.dirname(os.environ['PYMUMESSI_OUTDIR'])
        else:
            self.logdir = os.path.dirname(pymumessi.__path__[0])
            
        if not os.path.exists(self.logdir):
            os.makedirs(self.logdir, mode=0o0777)
            os.chmod(self.logdir, 0o0777)

        # deal with timezones in timestamped file paths
        # overload the default timestamp converter function
        # to one that is timezone-aware
        self.timezone = timezone
        if self.timezone is None:
            self.converter = lambda x: datetime.datetime.fromtimestamp(
                x, _logging_timezone)
        else:
            self.timezone = _get_timezone(self.timezone)
            self.converter = lambda x: datetime.datetime.fromtimestamp(
                x, self.timezone)

        self.day_path = self.converter(time.time()).strftime('%Y%m%d')
        self.log_colors = log_colors

    def setup_child_loggers(self, targets, alg_name,
                            extra_field_dict=None, custom_path=False, dir_suffix=None):
        """Sets up a custom logging adaptor for a target/algorithm pair.

        Use this to setup a flock of logging adaptors simultaneously so they will output
        to the same place (resolves issue where results are spread out over
        several folders)

        Args:
            targets: an iterable collection of your target objects (like SQUIDs)

            alg_name: this will be used to keep track of the loggers and will
                appear in the logging messages. It is required in order to label
                and identify the logger objects.

            extra_field_dict: A dictionary with additional information to be
                included in EVERY log message called through this child_logger.
                They will appear at the end of the message in the format
                "| {key}:{value}"
                NOTE: To include additional information in just one message call,
                    just use the 'extra' argument:
                        child_logger.info('log message',extra={'extra_field_key':'extra_field_value'})

            custom_path: str, default=False
                User directed path to directory within which to store logging files.
                (this is typically automatically generated).

            dir_suffix: str, default=None
                This will be appended to the automatically generated output_directory as:
                output_directory_dir_suffix. dir_suffix is NOT appended to custom_path.
                If custom_path is specified, dir_suffix is ignored (which is crucial for not breaking things).
                If not specified, this defaults to the global value, as set using
                `set_logging_dir_suffix`.
        """

        extra_fields = copy.deepcopy(extra_field_dict or {})

        extra_fields['macro'] = alg_name
        ut_time = self.converter(time.time()).strftime('%Y%m%d_%H%M%S')
        if custom_path:
            alg_dir = custom_path

        else:
            # # fall back to global default if not supplied
            # if dir_suffix is None:
            #     dir_suffix = _logging_dir_suffix

            if dir_suffix is None:
                alg_dir = os.path.join(self.logdir, self.day_path,
                                        "{0}_{1}".format(ut_time, alg_name))
            else:
                alg_dir = os.path.join(self.logdir, self.day_path,
                                        "{0}_{1}_{2}".format(ut_time, alg_name, dir_suffix))

        # Fix permissions for the day-path:
        try:
            day_path = os.path.join(self.logdir, self.day_path)
            os.makedirs(day_path, mode=0o0777)
            os.chmod(day_path, 0o0777)
        except OSError:
            # path already exists
            pass

        for target in targets:
            extra_fields = copy.deepcopy(extra_fields)
            # This deepcopy is necessary to keep all log messages from displaying
            # the reduced_target name of the last one configured

            # reduced_target = reduce_target_name(target)
            reduced_target = 'test'
            extra_fields['target'] = reduced_target

            log_path = os.path.join(alg_dir, 'logs')

            try:
                os.makedirs(log_path, mode=0o0777)
                os.chmod(log_path, 0o0777)
            except OSError:
                # path already exists
                pass

            # Remove special characters from the reduced_target
            reduced_target_savepath = reduced_target.replace('(', '_').replace(')', '')
            log_name = '{0}/{1}.txt'.format(log_path, reduced_target_savepath)
            child_file_logger = logging.FileHandler(log_name)
            child_file_logger.set_name('file_{0}_{1}'.format(alg_name,
                                                             reduced_target))

            child_formatter = DfmuxFormatter(self.BASIC_FORMAT,
                                             extra_fields=extra_fields,
                                             log_colors=self.log_colors,
                                             timezone=self.timezone)
            child_file_logger.setFormatter(child_formatter)
            child_logger = logging.getLogger('.{0}_{1}'.format(alg_name,
                                                               reduced_target))
            
            # remove stale handlers:
            for hdlr in child_logger.handlers:
                child_logger.removeHandler(hdlr)

            child_logger.addHandler(child_file_logger)
            child_logger.debug('Logging Adaptor is now active')
        return str(alg_dir)

    def get_child_logger(self, target, alg_name, setup=True,
                         extra_field_dict=None):
        """Returns a custom logging adaptor for a target-algorithm pair.

        Any calls to the returned ChildLogger will include
        persistant additional fields for the Macro in which the child_logger was
        instantiated, and the target object being operated on within that macro.
        This information, in addition to being handed up to the MainLogger handlers
        is saved in separate .txt files with a directory structure:
        PYDFMUX_LOG_DIR -> Date -> Macro Title -> Time_Object
        Such as:
        pydfmux_logs/2014_7_28/squid_vphi/64204.0_IceBoard(iceboard0017.local).MGMEZZ04(1,001).SQUIDController(None).SQUIDModule(1).SQUID.txt

        2014-12-10: changing this so that it is more reasonable
        pydfmux_out/2014_7_28/squid_vphi_64204/iceboard0017_Mezz1_Sq1.pkl

        Args:
            target: Ideally the target object should be a hardware-map type object, but
                it really only needs a sensible .__repr__() return

            alg_name: This will be used to keep track of the logger and will
                appear in the logging messages. It is required in order to label
                and identify the logger objects.

            setup: If you have already set up all of your logging adaptors using
                setup_child_loggers (so they all output to the same place)
                then set this to False, it will simply find and return you the
                correct one.

            extra_field_dict: A dictionary with additional information to be included
                in EVERY log message called through this child_logger. They will
                appear at the end of the message in the format "| {key}:{value}"
                NOTE: To include additional information in just one message call,
                    just use the 'extra' argument (child_logger.info('log message',extra={'extra_field_key':'extra_field_value'}))

        Output:
            Child Logging object which may be used normally to log events to the
            MainLogger handler as well as individual logfiles for algorithm runs by
            target and algorithm title.
        """

        extra_fields = copy.deepcopy(extra_field_dict or {})
        extra_fields['macro'] = alg_name

        # reduced_target = reduce_target_name(target)
        reduced_target = 'test'
        extra_fields['target'] = reduced_target

        if setup:
            self.setup_child_loggers(targets=[target], alg_name=alg_name,
                                    extra_field_dict=extra_fields)
        childlogger = logging.getLogger('.{0}_{1}'.format(alg_name, reduced_target))
        if not childlogger.handlers:
            warnings.warn("Returning a ChildLogger object with no handlers. "
                          "If you used setup_child_loggers(), the alg_name may "
                          "not match the one given then. "
                          "Alternatively, get_child_logger() can be called on its"
                          "own with 'setup=True'")
        return childlogger

    @staticmethod
    def set_console_level(level):
        """Set the threshold of the console logger (which prints to screen)

         By default the console logger is set to 'INFO' (level 20). Use this
         function to set to another level, such as 'DEBUG' (Level 10) or WARNING (level 20), etc

         Levels can be accessed from the logging module, and their numbers
         are reproduced here:

            logging.DEBUG    : 10
            logging.INFO     : 20
            logging.WARNING  : 30
            Logging.ERROR    : 40
            logging.CRITICAL : 50

         This function will also accept the string versions
         ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']

         Args:
            level: Should be an integer from 0-50 or the corresponding string.
                The strings and levels have been been reproduced above.
                By default this is set to INFO (20) when the logging manager is
                instantiated.
            """
        leveldict = {
            'DEBUG':     10,
            'INFO':      20,
            'NOTICE' :   25,
            'WARNING':   30,
            'WARN':      30,
            'ERROR':     40,
            'CRITICAL':  50
        }
        logger = logging.getLogger()

        if isinstance(level, str):
            level = level.upper()
            if level in leveldict:
                level = leveldict[level]
            else:
                raise TypeError('level argument not a recognized string or level number. Consult docstring')
        for handler in logger.handlers:
            if handler.name == 'console':
                handler.setLevel(level)
            if handler.name == 'file_console':
                handler.setLevel(level)
        logger.log(level, 'Console logging output level switched to {0}'.format(level))


class DfmuxFormatter(logging.Formatter):
    """Colorizes output to the console, and includes extra fields.

    The colorization portions were largely scavenged from colorlog.py
    (https://pypi.python.org/pypi/colorlog/2.3.1)

    Args:
        format: The base format string. In the DFMUX Logging Utils this
            looks a bit like: '%(asctime)s | %(levelname)s | %(filename)s | %(funcName)s | %(message)s'
            But is an explicit requirement for the class (no default)

        datefmt: The date formatting to apply to asctime.
            Default: '%Y-%m-%d %H:%M:%S'

        log_colors: A dictionary assigning logging levels specific coloring. If
            none is provided, by default it uses:
             {'DEBUG':    'cyan',
              'INFO':     'purple',
              'NOTICE':   'green',
              'WARNING':  'yellow',
              'ERROR':    'red',
              'CRITICAL': 'bold_red'}

        extra_fields: A dictionary with extra fields to add to the logging message
            and format string. Usually called from within the get_child_logger
            instantiation. Default: {}

        timezone: the timezone to be used for data output directory timestamps
            and log file entries.  Can be 'UTC', 'local', a string timezone name
            or `datetime.tzinfo` instance. If None, the global pydfmux default
            is used.
    """

    default_log_colors = {
        'DEBUG':    'cyan',
        'INFO':     'purple',
        'NOTICE' :  'green',
        'WARNING':  'yellow',
        'ERROR':    'red',
        'CRITICAL': 'bold_red',
    }

    def __init__(self, format, datefmt='%Y-%m-%d %H:%M:%S %Z',
                 log_colors=None, extra_fields={}, timezone=None):
        """Initialization of the DfmuxFormatter subclass -- builds escape codes
        and colors.
        """
        self.extra_fields = extra_fields
        if not log_colors:  # default log coloring
            log_colors = {
                'DEBUG':    'cyan',
                'INFO':     'purple',
                'NOTICE':   'green',
                'WARNING':  'yellow',
                'ERROR':    'red',
                'CRITICAL': 'bold_red',
            }
        self.log_colors = log_colors
        # Returns escape codes from format codes
        self.esc = lambda *x: '\033[' + ';'.join(x) + 'm'

        # The initial list of escape codes
        self.escape_codes = {
            'reset': self.esc('39', '49', '0'),
            'bold': self.esc('01'),
        }

        # The color names
        self.colors = [
            'black',
            'red',
            'green',
            'yellow',
            'blue',
            'purple',
            'cyan',
            'white'
        ]

        # Create foreground and background colors...
        for lcode, lname in [('3', ''), ('4', 'bg_')]:
            # ...with the list of colors...
            for code, name in enumerate(self.colors):
                code = str(code)
                # ...and both normal and bold versions of each color
                self.escape_codes[lname + name] = self.esc(lcode + code)
                self.escape_codes[lname + "bold_" + name] = self.esc(lcode + code, "01")

        if sys.version_info > (2, 7):
            super(DfmuxFormatter, self).__init__(format, datefmt)
        else:
            logging.Formatter.__init__(self, format, datefmt)

        # deal with timezones in file timestamps
        # overload the default timestamp converter function
        # to one that is timezone-aware
        self.timezone = timezone
        if self.timezone is None:
            self.converter = lambda x: datetime.datetime.fromtimestamp(
                x, _logging_timezone)
        else:
            self.timezone = _get_timezone(self.timezone)
            self.converter = lambda x: datetime.datetime.fromtimestamp(
                x, self.timezone)

    def formatTime(self, record, datefmt=None):
        """Overridden baseclass method for formatting timestamps with timezones."""

        ct = self.converter(record.created)

        if datefmt:
            s = ct.strftime(datefmt).strip()
        else:
            s = ct.strftime('%Y-%m-%d %H:M:%S.{} %z').strip().format(record.msecs)
        return s

    def format(self, record):
        """Overridden baseclass method that applies the escape codes and
        adds the custom fields to the message format (._fmt) and a copy of the
        record object, and then calls the baseclass method to build the message
        itself."""

        # Move the extra fields into the record used by upstream handlers
        record.__dict__.update(self.extra_fields)

        # copy the record object so we can manipulate it further without polluting
        # any upstream logging handlers
        record_copy = copy.copy(record)
        if record_copy.levelname == 'Level 25':
            record_copy.levelname = 'NOTICE'

        # Format the message
        # Save the original format configured by the user
        # when the logger formatter was instantiated
        format_orig = self._fmt

        # skip_list is used to filter additional fields in a log message.
        # We do this before adding the colors to avoid adding the escape
        # codes as additional fields
        # To skip are all attributes listed in
        # http://docs.python.org/library/logging.html#logrecord-attributes
        # plus exc_text, which is only found in the logging module source,
        # and id, which is prohibited by the GELF format.

        skip_list = (
            'args', 'asctime', 'created', 'exc_info',  'exc_text', 'filename',
            'funcName', 'id', 'levelname', 'levelno', 'lineno', 'module',
            'msecs', 'msecs', 'message', 'msg', 'name', 'pathname', 'process',
            'processName', 'relativeCreated', 'thread', 'threadName')

        for key, value in list(record_copy.__dict__.items()):
            if key not in skip_list and not key.startswith('_') \
                    and key not in self.escape_codes:
                if key == 'target':
                    self._fmt = self._fmt.replace('| %(funcName)s |',
                                                  '| %(funcName)s | %(target)s |')
                elif key == 'macro':
                    self._fmt = self._fmt.replace('| %(filename)s |',
                                                  '| %(macro)s | %(filename)s |')
                elif key in self.extra_fields.keys():
                    if '%(target)s' in self._fmt:
                        self._fmt = self._fmt.replace('| %(target)s |',
                                                      '| %({0})s | %(target)s |'.format(key))
                    else:
                        self._fmt = self._fmt.replace('| %(funcName)s |',
                                                      '| %(funcName)s | %({0})s |'.format(key))
                else:
                    self._fmt = self._fmt + ' | {0}: %({0})s'.format(key)

            record_copy.__dict__.update(self.escape_codes)
            # If we recognise the level name,
            # add the levels color as `log_color`
            if record_copy.levelname in self.log_colors:
                color = self.log_colors[record_copy.levelname]
                record_copy.log_color = self.escape_codes[color]
            else:
                record_copy.log_color = ""

        if sys.version_info > (2, 7):
            message = super(DfmuxFormatter, self).format(record_copy)
        else:
            message = logging.Formatter.format(self, record_copy)

        if 'log_color' in self._fmt:
            # Add a reset code to the end of the message
            # (if it wasn't explicitly added in format str)
            # But only if escape codes were used in the first place
            if not message.endswith(self.escape_codes['reset']):
                message += self.escape_codes['reset']

        # Restore the original format configured by the user
        self._fmt = format_orig
        return message
