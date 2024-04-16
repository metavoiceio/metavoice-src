import logging
import os
import sys

from dotenv import load_dotenv
from posthog import Posthog

from fam.telemetry import TelemetryClient, TelemetryEvent

load_dotenv()
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, handlers=[logging.StreamHandler(sys.stdout), logging.StreamHandler(sys.stderr)])


class PosthogClient(TelemetryClient):
    def __init__(self):
        self._posthog = Posthog(
            project_api_key="phc_tk7IUlV7Q7lEa9LNbXxyC1sMWlCqiW6DkHyhJrbWMCS", host="https://eu.posthog.com"
        )

        if not os.getenv("ANONYMIZED_TELEMETRY", True) or "pytest" in sys.modules:
            self._posthog.disabled = True
            logger.info("Anonymized telemetry disabled. See fam/telemetry/README.md for more information.")
        else:
            logger.info("Anonymized telemetry enabled. See fam/telemetry/README.md for more information.")

        posthog_logger = logging.getLogger("posthog")
        posthog_logger.disabled = True  # Silence posthog's logging

        super().__init__()

    def capture(self, event: TelemetryEvent) -> None:
        try:
            self._posthog.capture(
                self.user_id,
                event.name,
                {**event.properties},
            )
        except Exception as e:
            logger.error(f"Failed to send telemetry event {event.name}: {e}")
