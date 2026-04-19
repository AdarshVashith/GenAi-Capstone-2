"""CLI smoke test for the Farm Advisory agent."""

import argparse
import logging

from agent.farm_agent import run_farm_agent

logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a quick farm advisory prediction.")
    parser.add_argument("--area", default="India", help="Geographic area (default: India)")
    parser.add_argument("--crop", default="Wheat", help="Crop type (default: Wheat)")
    parser.add_argument("--temp", type=float, default=25.0, help="Avg temperature (default: 25.0)")
    parser.add_argument("--rainfall", type=float, default=800.0, help="Rainfall mm/year (default: 800)")
    parser.add_argument("--pesticides", type=float, default=50.0, help="Pesticide usage in tonnes (default: 50)")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    farm_data = {
        "Area": args.area,
        "Item": args.crop,
        "avg_temp": args.temp,
        "average_rain_fall_mm_per_year": args.rainfall,
        "pesticides_tonnes": args.pesticides,
    }
    try:
        final_state = run_farm_agent(farm_data)
    except Exception as exc:
        logger.error("Unable to run farm agent: %s", exc)
        return

    logger.info("Yield Prediction: %s", final_state["yield_prediction"])
    logger.info("Report:\n%s", final_state["final_report"])


if __name__ == "__main__":
    main()
