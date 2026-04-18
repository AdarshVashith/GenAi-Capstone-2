"""CLI smoke test for the Farm Advisory agent."""

from agent.farm_agent import run_farm_agent


def main() -> None:
    farm_data = {
        "Area": "India",
        "Item": "Wheat",
        "avg_temp": 25.0,
        "average_rain_fall_mm_per_year": 800.0,
        "pesticides_tonnes": 50.0,
    }
    try:
        final_state = run_farm_agent(farm_data)
    except Exception as exc:
        print(f"Unable to run farm agent: {exc}")
        return

    print("Yield Prediction:", final_state["yield_prediction"])
    print()
    print(final_state["final_report"])


if __name__ == "__main__":
    main()
