"""
Matplotlib-based timeseries plotting (DEPRECATED).

Plotting is now routed through rendering/plotly_renderer.py.
This module is kept as a legacy fallback.
"""

import os
from datetime import datetime

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np

from .store import DataEntry


def plot_timeseries(
    entries: list[DataEntry],
    title: str = "",
    filename: str = "",
) -> str:
    """Render timeseries to a PNG file.

    Supports overlaying multiple timeseries. Vector entries (n, 3) are
    plotted as separate x/y/z component lines.

    Args:
        entries: List of DataEntry objects to plot.
        title: Plot title (auto-generated if empty).
        filename: Output filename (auto-generated with timestamp if empty).

    Returns:
        Absolute path to the saved PNG file.
    """
    if not entries:
        raise ValueError("No entries to plot")

    fig, ax = plt.subplots(figsize=(12, 5))

    units_set = set()

    for entry in entries:
        # Convert datetime64[ns] to matplotlib-compatible dates
        times = entry.time.astype("datetime64[ms]").astype(datetime)

        if entry.values.ndim == 2 and entry.values.shape[1] > 1:
            # Vector: plot each component
            component_labels = ["x", "y", "z"]
            for col in range(entry.values.shape[1]):
                comp = component_labels[col] if col < 3 else str(col)
                ax.plot(
                    times,
                    entry.values[:, col],
                    label=f"{entry.label}.{comp}",
                    linewidth=0.7,
                )
        else:
            ax.plot(
                times,
                entry.values,
                label=entry.label,
                linewidth=0.7,
            )

        if entry.units:
            units_set.add(entry.units)

    # Axis labels
    ax.set_xlabel("Time (UTC)")
    if units_set:
        ax.set_ylabel(" / ".join(sorted(units_set)))

    # Title
    if not title:
        title = ", ".join(e.label for e in entries)
    ax.set_title(title)

    # Legend
    ax.legend(fontsize="small", loc="best")

    # Time axis formatting
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d\n%H:%M"))
    fig.autofmt_xdate()
    ax.grid(True, alpha=0.3)

    fig.tight_layout()

    # Generate filename if not provided
    if not filename:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"plot_{ts}.png"
    if not filename.endswith(".png"):
        filename += ".png"

    filepath = os.path.abspath(filename)
    fig.savefig(filepath, dpi=150)
    plt.close(fig)

    return filepath
