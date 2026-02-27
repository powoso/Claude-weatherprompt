"""Alert system for new opportunities via email, Slack, and Discord."""

from __future__ import annotations

import json
import logging
import smtplib
from email.mime.text import MIMEText

import httpx

from weather_edge.edge.detector import Opportunity

logger = logging.getLogger("weather_edge.alerts.notifier")


def send_alerts(
    opportunities: list[Opportunity],
    config: dict,
    alert_type: str = "new_opportunity",
) -> None:
    """Send alerts for detected opportunities through all enabled channels."""
    alert_config = config.get("alerts", {})
    if not alert_config.get("enabled", False):
        return

    if not opportunities:
        return

    message = _format_alert_message(opportunities, alert_type)

    if alert_config.get("email", {}).get("enabled"):
        _send_email(message, alert_config["email"])

    if alert_config.get("slack", {}).get("enabled"):
        _send_slack(message, alert_config["slack"])

    if alert_config.get("discord", {}).get("enabled"):
        _send_discord(message, alert_config["discord"])


def _format_alert_message(
    opportunities: list[Opportunity],
    alert_type: str,
) -> str:
    """Format opportunities into a readable alert message."""
    lines = []
    if alert_type == "new_opportunity":
        lines.append("== NEW WEATHER MARKET OPPORTUNITIES ==\n")
    elif alert_type == "edge_widened":
        lines.append("== EDGE WIDENED ON EXISTING OPPORTUNITIES ==\n")

    for opp in opportunities:
        lines.append(f"  Contract: {opp.title}")
        lines.append(f"  Source: {opp.source}")
        lines.append(f"  Model Prob: {opp.model_prob:.1%}")
        lines.append(f"  Market Price: {opp.market_prob:.1%}")
        lines.append(f"  Edge: {opp.edge:+.1%}")
        lines.append(f"  EV: {opp.expected_value:+.2f}")
        lines.append(f"  Kelly Bet: ${opp.suggested_bet_size:.2f}")
        lines.append(f"  CI: [{opp.confidence_interval[0]:.1%}, {opp.confidence_interval[1]:.1%}]")
        lines.append("")

    lines.append(f"Total opportunities: {len(opportunities)}")
    return "\n".join(lines)


def _send_email(message: str, email_config: dict) -> None:
    """Send alert via email/SMTP."""
    try:
        msg = MIMEText(message)
        msg["Subject"] = "Weather Edge Alert"
        msg["From"] = email_config["sender"]
        msg["To"] = ", ".join(email_config["recipients"])

        with smtplib.SMTP(email_config["smtp_server"], email_config["smtp_port"]) as server:
            server.starttls()
            server.login(email_config["sender"], email_config["password"])
            server.send_message(msg)
        logger.info("Email alert sent to %s", email_config["recipients"])
    except Exception:
        logger.exception("Failed to send email alert")


def _send_slack(message: str, slack_config: dict) -> None:
    """Send alert via Slack webhook."""
    webhook_url = slack_config.get("webhook_url", "")
    if not webhook_url:
        return
    try:
        payload = {"text": f"```\n{message}\n```"}
        resp = httpx.post(webhook_url, json=payload, timeout=10)
        resp.raise_for_status()
        logger.info("Slack alert sent")
    except Exception:
        logger.exception("Failed to send Slack alert")


def _send_discord(message: str, discord_config: dict) -> None:
    """Send alert via Discord webhook."""
    webhook_url = discord_config.get("webhook_url", "")
    if not webhook_url:
        return
    try:
        payload = {"content": f"```\n{message}\n```"}
        resp = httpx.post(webhook_url, json=payload, timeout=10)
        resp.raise_for_status()
        logger.info("Discord alert sent")
    except Exception:
        logger.exception("Failed to send Discord alert")
