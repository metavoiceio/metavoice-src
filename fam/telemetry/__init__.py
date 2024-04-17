import abc
import os
import uuid
from abc import abstractmethod
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class TelemetryEvent:
    name: str
    properties: dict


class TelemetryClient(abc.ABC):
    USER_ID_PATH = str(Path.home() / ".cache" / "metavoice" / "telemetry_user_id")
    UNKNOWN_USER_ID = "UNKNOWN"
    _curr_user_id = None

    @abstractmethod
    def capture(self, event: TelemetryEvent) -> None:
        pass

    @property
    def user_id(self) -> str:
        if self._curr_user_id:
            return self._curr_user_id

        # File access may fail due to permissions or other reasons. We don't want to
        # crash so we catch all exceptions.
        try:
            if not os.path.exists(self.USER_ID_PATH):
                os.makedirs(os.path.dirname(self.USER_ID_PATH), exist_ok=True)
                with open(self.USER_ID_PATH, "w") as f:
                    new_user_id = str(uuid.uuid4())
                    f.write(new_user_id)
                self._curr_user_id = new_user_id
            else:
                with open(self.USER_ID_PATH, "r") as f:
                    self._curr_user_id = f.read()
        except Exception:
            self._curr_user_id = self.UNKNOWN_USER_ID
        return self._curr_user_id
