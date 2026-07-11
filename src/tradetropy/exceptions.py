class TradetropyError(Exception):
    """
    Base class for all custom tradetropy exceptions.

    All tradetropy-specific exceptions inherit from this class for consistent
    error handling and filtering across the framework.
    """


class ConfigError(TradetropyError):
    """
    Invalid configuration or parameters.

    Raised when configuration parameters are invalid, including incorrect
    symbols, unsupported timeframes, or incorrect data types.
    """


class ConnectionError(TradetropyError):
    """
    Connection failure to exchange or data source.

    Raised when the broker cannot establish or maintain a connection to the
    exchange API or data provider.
    """


class DataError(TradetropyError):
    """
    Invalid data structure or content.

    Raised when data is malformed, missing required columns, has incorrect
    shapes, or contains invalid timestamps.
    """


class ColumnNotFoundError(DataError):
    """
    Column not found in the data proxy.

    Raised when accessing a column that does not exist in the data provider.
    """


class TradingError(TradetropyError):
    """
    Trading operation failure.

    Raised when order execution fails, required price data is unavailable,
    or the engine is not ready to process trades.
    """


class MarginCallError(TradingError):
    """
    Broker executed a stop out due to margin call.

    Raised when the account equity falls below the required maintenance margin
    and the broker liquidates open positions to restore the account state.
    """


class PatternSyntaxError(TradetropyError):
    """
    Syntax error in the pattern DSL.

    Raised when the pattern definition language contains invalid syntax.
    """


class TimeConditionError(PatternSyntaxError):
    """
    Error in temporal conditions of the pattern DSL.

    Raised when time-based conditions in a pattern are invalid or
    cannot be evaluated.
    """


class StopEngine(TradetropyError):
    """
    Raised by the strategy to stop the engine gracefully.

    Used as a signal from the strategy to halt execution without treating
    it as an error condition.

    Args:
        reason (str): Optional explanation for stopping the engine.
    """
    def __init__(self, reason: str = ''):
        self.reason = reason
        super().__init__(reason)


class NotReleasedError(TradetropyError):
    """
    Raised when attempting to use a module that has not been released yet.

    Used in dummies for features that will be released once a certain
    number of monthly sponsors is reached.

    Args:
        module_name: Name of the unreleased module.
        sponsors_needed: Number of sponsors required for release.
    """
    def __init__(self, module_name: str, sponsors_needed: int):
        self.module_name = module_name
        self.sponsors_needed = sponsors_needed
        msg = (
            f"'{module_name}' will be released when {sponsors_needed} "
            f"monthly sponsors are reached."
        )
        super().__init__(msg)
