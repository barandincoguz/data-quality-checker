"""Typed failures used at command and recovery boundaries."""


class DQCheckError(RuntimeError):
    """Base error safe to render at the CLI boundary."""


class ConfigurationError(DQCheckError):
    pass


class ContractError(DQCheckError):
    pass


class IntegrityError(DQCheckError):
    pass


class FingerprintMismatch(IntegrityError):
    pass


class LockUnavailable(DQCheckError):
    pass


class VersionConflict(DQCheckError):
    pass


class UnsafeArchive(DQCheckError):
    pass


class GateBlocked(DQCheckError):
    pass
