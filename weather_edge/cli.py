"""CLI entry point for weather-edge."""

from __future__ import annotations

import sys
from pathlib import Path

import click
from dotenv import load_dotenv

load_dotenv()


@click.group()
@click.option("--config", "-c", default=None, help="Path to config.yaml")
@click.pass_context
def main(ctx, config):
    """Weather Prediction Market Edge Finder."""
    from weather_edge.utils.config import load_config
    from weather_edge.utils.logging import setup_logging

    cfg = load_config(config)
    setup_logging(cfg)
    ctx.ensure_object(dict)
    ctx.obj["config"] = cfg


@main.command()
@click.pass_context
def ingest(ctx):
    """Download and cache all historical data (weather, hurricanes, ENSO)."""
    from weather_edge.pipeline import run_ingest
    run_ingest(ctx.obj["config"])
    click.echo("Ingest complete.")


@main.command()
@click.pass_context
def scan(ctx):
    """Scan prediction markets, evaluate contracts, and detect +EV opportunities."""
    from weather_edge.pipeline import run_scan
    opportunities = run_scan(ctx.obj["config"])
    if opportunities:
        click.echo(f"\nFound {len(opportunities)} opportunities:")
        for opp in opportunities:
            click.echo(
                f"  [{opp.source}] {opp.title[:60]}\n"
                f"    Model: {opp.model_prob:.1%} | Market: {opp.market_prob:.1%} | "
                f"Edge: {opp.edge:+.1%} | Kelly bet: ${opp.suggested_bet_size:.2f}"
            )
    else:
        click.echo("No +EV opportunities found at current thresholds.")


@main.command()
@click.pass_context
def update(ctx):
    """Full pipeline: fetch forecasts, rerun models, update probabilities."""
    from weather_edge.pipeline import run_update
    run_update(ctx.obj["config"])
    click.echo("Update complete.")


@main.command()
@click.argument("city")
@click.option("--metric", "-m", default="temperature_2m_max", help="Weather metric")
@click.option("--threshold", "-t", type=float, default=None, help="Threshold value")
@click.option("--month", type=int, default=None, help="Month (1-12)")
@click.pass_context
def base_rates(ctx, city, metric, threshold, month):
    """Compute historical base rates for a city/metric/threshold."""
    from weather_edge.pipeline import run_base_rates
    result = run_base_rates(ctx.obj["config"], city, metric, threshold, month)

    if "error" in result:
        click.echo(f"Error: {result['error']}")
        return

    click.echo(f"\nBase rates for {result['city']} ({metric}):")

    if "exceedance" in result:
        exc = result["exceedance"]
        click.echo(f"  Threshold: {threshold}")
        click.echo(f"  Daily exceedance rate: {exc['base_rate_daily']:.2%}")
        click.echo(f"  Yearly exceedance rate: {exc['base_rate_yearly']:.2%}")
        click.echo(f"  ({exc.get('exceed_years', '?')}/{exc.get('total_years', '?')} years)")

    if "distribution_fit" in result:
        fit = result["distribution_fit"]
        click.echo(f"\n  Distribution: {fit['distribution']}")
        click.echo(f"  Mean: {fit['mean']:.1f}  Std: {fit['std']:.1f}")


@main.command()
@click.option("--host", default=None, help="Host to bind to")
@click.option("--port", "-p", type=int, default=None, help="Port to bind to")
@click.option("--debug", is_flag=True, help="Enable debug mode")
@click.pass_context
def dashboard(ctx, host, port, debug):
    """Launch the web dashboard."""
    from weather_edge.webapp.app import create_app
    cfg = ctx.obj["config"]
    host = host or cfg.get("dashboard", {}).get("host", "0.0.0.0")
    port = port or cfg.get("dashboard", {}).get("port", 5000)
    flask_app = create_app()
    click.echo(f"Launching dashboard at http://{host}:{port}")
    flask_app.run(host=host, port=port, debug=debug)


@main.command()
@click.pass_context
def cron(ctx):
    """Run the scheduled update loop (blocking)."""
    import schedule
    import time

    config = ctx.obj["config"]
    from weather_edge.pipeline import run_scan, run_update

    # Schedule tasks
    schedule.every(6).hours.do(run_update, config)
    schedule.every(1).hours.do(run_scan, config)

    click.echo("Cron scheduler started. Press Ctrl+C to stop.")
    click.echo("  - Full update: every 6 hours")
    click.echo("  - Market scan: every 1 hour")

    # Run initial scan
    run_scan(config)

    while True:
        schedule.run_pending()
        time.sleep(60)


if __name__ == "__main__":
    main()
