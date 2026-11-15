"""Outbound calling through Twilio Voice.

Google has no consumer calling API, so JARVIS places calls through
Twilio. For a voice-agent experience, point Twilio at a webhook that
bridges audio to a speech pipeline (STT -> agent -> TTS); that bridge is
a later milestone and is not implemented here.
"""

from __future__ import annotations

from xml.sax.saxutils import escape

from .base import Connector


class TwilioCallingConnector(Connector):
    name = "calling"
    required_settings = ("twilio_account_sid", "twilio_auth_token", "twilio_from_number")

    @property
    def api(self) -> str:
        sid = self.settings.twilio_account_sid
        return f"https://api.twilio.com/2010-04-01/Accounts/{sid}/Calls.json"

    def start_call(self, to: str, say: str) -> dict:
        """Start an outbound call that speaks `say` via TwiML, then hangs up.
        Write operation: requires credentials and actions enabled."""
        self.require_enabled()
        # Escape the spoken text: raw <, > or & would make the TwiML invalid.
        twiml = f"<Response><Say>{escape(say)}</Say></Response>"
        response = self._client.post(
            self.api,
            auth=(self.settings.twilio_account_sid, self.settings.twilio_auth_token),
            data={
                "To": to,
                "From": self.settings.twilio_from_number,
                "Twiml": twiml,
            },
        )
        response.raise_for_status()
        return response.json()

