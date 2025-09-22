"""Score reporting and comparison functionality."""

import logging
from pathlib import Path
from typing import List

import polars as pl

from ..scores import BaseScoringResult, MetricsConfig


class ScoreReport:
    """Generate comparison reports of judge performance across metrics."""

    def __init__(self, scores_dir: str | Path, metrics_config: MetricsConfig):
        """Initialize ScoreReport with scores directory and metrics configuration.

        Args:
            scores_dir: Path to the scores directory containing result JSON files
            metrics_config: MetricsConfig containing metric configurations with unique names
        """
        self.scores_dir = Path(scores_dir)
        self.metrics_config = metrics_config
        self.report_df: pl.DataFrame | None = None
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

    def load_single_results(
        self, unique_name: str, scorer_name: str
    ) -> List[BaseScoringResult]:
        """Load results for a specific metric configuration.

        Args:
            unique_name: Unique name from MetricConfig.get_unique_name()
            scorer_name: Name of the scorer (e.g., "accuracy", "alt_test")

        Returns:
            List[BaseScoringResult]: List of results for this specific configuration
        """
        results = []

        # Look in: scores/{scorer_name}/{unique_name}/
        target_dir = self.scores_dir / scorer_name / unique_name

        if target_dir.exists():
            for json_file in target_dir.glob("*_result.json"):
                try:
                    result = BaseScoringResult.load_state(str(json_file))
                    results.append(result)
                    self.logger.debug(f"Loaded result from {json_file}")
                except Exception as e:
                    self.logger.warning(f"Failed to load result from {json_file}: {e}")
                    continue
        else:
            self.logger.warning(f"Target directory does not exist: {target_dir}")

        self.logger.debug(
            f"Loaded {len(results)} results for {scorer_name}/{unique_name}"
        )
        return results

    def generate(self) -> pl.DataFrame:
        """Generate detailed report using MetricsConfig unique names as columns.

        Returns:
            pl.DataFrame: DataFrame with judges as rows and unique metric configurations as columns
        """
        judge_data = {}  # Collect all metrics per judge

        for metric_config in self.metrics_config.metrics:
            unique_name = metric_config.get_unique_name()
            scorer_name = metric_config.scorer.scorer_name

            # Load results for this specific config
            results = self.load_single_results(unique_name, scorer_name)

            for result in results:
                # Initialize judge entry if not exists
                if result.judge_id not in judge_data:
                    judge_data[result.judge_id] = {"judge_id": result.judge_id}

                # Add metric scores to this judge's row
                if scorer_name == "accuracy":
                    judge_data[result.judge_id][unique_name] = result.scores.get(
                        "accuracy"
                    )
                elif scorer_name == "alt_test":
                    # Two columns for alt_test: winning rate and advantage probability
                    winning_rates = result.scores.get("winning_rate", {})
                    winning_rate_02 = winning_rates.get(
                        "0.20", winning_rates.get("0.2")
                    )
                    advantage_prob = result.scores.get("advantage_probability")

                    judge_data[result.judge_id][f"{unique_name}_winning_rate"] = (
                        winning_rate_02
                    )
                    judge_data[result.judge_id][f"{unique_name}_advantage_prob"] = (
                        advantage_prob
                    )
                elif scorer_name == "cohens_kappa":
                    judge_data[result.judge_id][unique_name] = result.scores.get(
                        "kappa"
                    )
                elif scorer_name == "text_similarity":
                    judge_data[result.judge_id][unique_name] = result.scores.get(
                        "similarity"
                    )
                elif scorer_name == "semantic_similarity":
                    judge_data[result.judge_id][unique_name] = result.scores.get(
                        "semantic_similarity"
                    )

        if not judge_data:
            self.logger.warning("No valid score data found, returning empty DataFrame")
            return pl.DataFrame()

        # Convert judge_data dict to list of rows
        rows = list(judge_data.values())

        # Just create DataFrame directly - no grouping needed!
        self.report_df = pl.DataFrame(rows)

        self.logger.info(
            f"Generated detailed report with {len(self.report_df)} judges and {len(self.report_df.columns) - 1} metric configurations"
        )
        return self.report_df

    def print(self) -> None:
        """Generate and print the DataFrame report to console."""
        if self.report_df is None:
            self.generate()

        if self.report_df is None or len(self.report_df) == 0:
            print("No data available for report generation.")
            return

        print(f"\nScore Report:\n{self.report_df}")

    def save(self, filename: str, format: str = "html") -> str:
        """Save the comparison report to file in the scores directory.

        Args:
            filename: Filename for the report (will be saved in scores directory)
            format: Output format ("html", "csv", or "txt")

        Returns:
            str: The content that was saved to the file
        """
        if self.report_df is None:
            self.generate()

        if self.report_df is None or len(self.report_df) == 0:
            self.logger.warning("No data available to save")
            return ""

        # Build full path using scores directory
        output_path = self.scores_dir / filename

        if format == "html":
            content = self._to_html()
        elif format == "csv":
            content = self.report_df.write_csv()
        else:  # txt or default
            content = str(self.report_df)

        with open(output_path, "w") as f:
            f.write(content)

        self.logger.info(f"Report saved to {output_path} in {format} format")
        return content

    def _to_html(self) -> str:
        """Generate HTML table with highlighting for best performers.

        Returns:
            str: HTML table with styling and highlighting
        """
        if self.report_df is None or len(self.report_df) == 0:
            return "<p>No data available.</p>"

        html = """
        <style>
        .score-report-table {
            border-collapse: collapse;
            width: 100%;
            font-family: Arial, sans-serif;
        }
        .score-report-table th, .score-report-table td {
            border: 1px solid #ddd;
            padding: 8px;
            text-align: center;
        }
        .score-report-table th {
            background-color: #f2f2f2;
            font-weight: bold;
        }
        .best-score {
            background-color: lightgreen !important;
            font-weight: bold;
        }
        </style>
        """

        html += '<table class="score-report-table">\n'
        html += "<tr><th>Judge</th>"

        # Header - skip judge_id column
        metric_columns = [col for col in self.report_df.columns if col != "judge_id"]
        for col in metric_columns:
            html += f"<th>{col.replace('_', ' ').title()}</th>"
        html += "</tr>\n"

        # Rows with highlighting
        for row in self.report_df.iter_rows(named=True):
            html += "<tr>"
            html += f"<td>{row['judge_id']}</td>"

            for col in metric_columns:
                value = row[col]
                if value is not None and not pl.datatypes.Null().is_(value):
                    # Find if this is the best score in this column
                    is_best = value == self.report_df[col].max()
                    css_class = "best-score" if is_best else ""
                    html += f'<td class="{css_class}">{value:.3f}</td>'
                else:
                    html += "<td>-</td>"
            html += "</tr>\n"

        html += "</table>"
        return html
