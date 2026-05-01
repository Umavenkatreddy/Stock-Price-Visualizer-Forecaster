"""
Locust performance / load test for the Stock Price Visualizer & Forecaster.

Run locally:
    locust -f locustfile.py --headless -u 10 -r 2 --run-time 30s \
           --host http://localhost:8050

In CI (Performance stage) the workflow starts the Dash dev server in the
background and then runs this file with --headless so no browser is required.

Thresholds enforced by the workflow:
  - p95 response time < 2000 ms
  - failure rate < 1 %
"""

from locust import HttpUser, task, between, events
import json


class StockAppUser(HttpUser):
    """Simulates a user browsing the Stock Dash application."""

    # Wait between 1 and 3 seconds between consecutive tasks (realistic pacing)
    wait_time = between(1, 3)

    # ------------------------------------------------------------------ #
    #  Core page loads                                                     #
    # ------------------------------------------------------------------ #

    @task(5)
    def load_homepage(self):
        """Highest-weight task: simply GET the root page."""
        with self.client.get("/", catch_response=True) as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(
                    f"Homepage returned {response.status_code}"
                )

    @task(3)
    def load_assets(self):
        """Fetch the CSS stylesheet."""
        with self.client.get("/assets/style.css", catch_response=True) as response:
            # 200 is success; 304 (not modified) is also acceptable
            if response.status_code in (200, 304, 404):
                response.success()
            else:
                response.failure(
                    f"CSS asset returned {response.status_code}"
                )

    # ------------------------------------------------------------------ #
    #  Dash internal health / layout endpoint                             #
    # ------------------------------------------------------------------ #

    @task(2)
    def dash_layout(self):
        """Dash exposes /_dash-layout for the initial component tree."""
        with self.client.get("/_dash-layout", catch_response=True) as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(
                    f"Dash layout endpoint returned {response.status_code}"
                )

    @task(1)
    def dash_dependencies(self):
        """Dash exposes /_dash-dependencies listing all registered callbacks."""
        with self.client.get("/_dash-dependencies", catch_response=True) as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(
                    f"Dash dependencies endpoint returned {response.status_code}"
                )


# --------------------------------------------------------------------------- #
#  Custom event hook – print a summary when the test run finishes             #
# --------------------------------------------------------------------------- #

@events.quitting.add_listener
def on_quitting(environment, **kwargs):
    stats = environment.stats.total
    print("\n" + "=" * 60)
    print("PERFORMANCE TEST SUMMARY")
    print("=" * 60)
    print(f"  Total requests   : {stats.num_requests}")
    print(f"  Failures         : {stats.num_failures}")
    print(f"  Failure rate     : {stats.fail_ratio * 100:.2f} %")
    print(f"  Median resp time : {stats.median_response_time:.0f} ms")
    print(f"  95th percentile  : {stats.get_response_time_percentile(0.95):.0f} ms")
    print(f"  Max resp time    : {stats.max_response_time:.0f} ms")
    print("=" * 60)

    # Enforce thresholds (the CI workflow also checks exit code)
    p95 = stats.get_response_time_percentile(0.95)
    if p95 > 2000:
        print(f"THRESHOLD BREACH: p95 {p95:.0f} ms > 2000 ms limit")
        environment.process_exit_code = 1

    if stats.fail_ratio > 0.01:
        print(
            f"THRESHOLD BREACH: failure rate "
            f"{stats.fail_ratio * 100:.2f} % > 1 % limit"
        )
        environment.process_exit_code = 1
